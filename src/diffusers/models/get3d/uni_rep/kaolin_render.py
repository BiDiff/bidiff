import glob
import math
import copy
import os
import numpy as np
import json

import torch
from matplotlib import pyplot as plt
# from tutorial_common import COMMON_DATA_DIR

import kaolin as kal
from torchvision.utils import save_image

import nvdiffrast
# glctx = nvdiffrast.torch.RasterizeCudaContext(device='cuda')

def generate_pinhole_rays_dir(camera, height, width, device='cuda'):
    """Generate centered grid.
    
    This is a utility function for specular reflectance with spherical gaussian.
    """
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    pixel_x = pixel_x + 0.5  # scale and add bias to pixel center
    pixel_y = pixel_y + 0.5  # scale and add bias to pixel center

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    # Convert to NDC
    pixel_x = 2 * (pixel_x / width) - 1.0
    pixel_y = 2 * (pixel_y / height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

    return ray_dir[0].reshape(1, height, width, 3)

def base_render(camera, height, width, mesh, azimuth, elevation, amplitude, sharpness):
    """Base rendering function"""
    glctx = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_clip = camera.intrinsics.project(vertices_camera)
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
        vertices_camera, mesh.faces)
    faces_int = mesh.faces.int()
    rast = nvdiffrast.torch.rasterize(
        glctx, vertices_clip, faces_int,
        (height, width), grad_db=False)
    rast0 = torch.flip(rast[0], dims=(1,))
    hard_mask = rast0[:, :, :, -1:] != 0
    face_idx = (rast0[..., -1].long() - 1).contiguous()
    coords = nvdiffrast.torch.interpolate(
        vertices_camera, rast0, faces_int
    )[0]
    if mesh.has_or_can_compute_attribute('vertex_normals'):
        im_base_normals = nvdiffrast.torch.interpolate(
            mesh.vertex_normals, rast0, faces_int
        )[0]
    elif mesh.has_or_can_compute_attribute('normals') and mesh.has_attribute('face_normals_idx'):
        im_base_normals = nvdiffrast.torch.interpolate(
            mesh.normals, rast0, mesh.face_normals_idx.int()
        )[0]
    else:
        raise KeyError("mesh has no normal information")
    vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_clip)
    face_vertices_ndc = kal.ops.mesh.index_vertices_by_faces(
        vertices_ndc, mesh.faces
    )
    edges_dist0 = face_vertices_ndc[:, :, 1, :2] - face_vertices_ndc[:, :, 0, :2]
    edges_dist1 = face_vertices_ndc[:, :, 2, :2] - face_vertices_ndc[:, :, 0, :2]
    face_normal_sign = edges_dist0[..., 0] * edges_dist1[..., 1] - edges_dist1[..., 0] * edges_dist0[..., 1]

    im_normal_sign = torch.sign(face_normal_sign[0, face_idx])
    im_normal_sign[face_idx == -1] = 0.
    im_base_normals *= im_normal_sign.unsqueeze(-1)

    if mesh.uvs is not None:
        uv_map = nvdiffrast.torch.interpolate(
            mesh.uvs, rast0, mesh.face_uvs_idx.int()
        )[0] % 1.
        im_tangents = nvdiffrast.torch.interpolate(
            mesh.vertex_tangents, rast0, faces_int
        )[0]
        im_bitangents = torch.nn.functional.normalize(
            torch.cross(im_tangents, im_base_normals), dim=-1
        )
    im_material_idx = mesh.material_assignments[face_idx]
    im_material_idx[face_idx == -1] = -1
    albedo = torch.zeros((1, height, width, 3), device='cuda')
    spec_albedo = torch.zeros((1, height, width, 3), device='cuda')
    im_world_normals = torch.zeros((1, height, width, 3), device='cuda')
    im_roughness = torch.zeros((1, height, width, 1), device='cuda')
    for i, material in enumerate(mesh.materials):
        mask = im_material_idx == i
        if mesh.uvs is not None:
            _texcoords = uv_map[mask]
        if material.normals_texture is None:
            im_world_normals[mask] = im_base_normals[mask]
        else:
            if _texcoords.shape[0] > 0:
                perturbation_normal = nvdiffrast.torch.texture(
                    material.normals_texture.unsqueeze(0),
                    _texcoords.reshape(1, 1, -1, 2).contiguous(),
                    filter_mode='linear'
                )
                shading_normals = torch.nn.functional.normalize(
                    im_tangents[mask] * perturbation_normal[..., :1]
                    - im_bitangents[mask] * perturbation_normal[..., 1:2]
                    + im_base_normals[mask] * perturbation_normal[..., 2:3],
                    dim=-1
                )
                im_world_normals[mask] = shading_normals

        if material.diffuse_texture is None:
            if material.diffuse_color is not None:
                albedo[mask] = material.diffuse_color.unsqueeze(0)
        else:
            if _texcoords.shape[0] > 0:
                pixel_val = nvdiffrast.torch.texture(
                    material.diffuse_texture.unsqueeze(0),
                    _texcoords.reshape(1, 1, -1, 2).contiguous(),
                    filter_mode='linear'
                )
                albedo[mask] = pixel_val[0, 0]

        if material.is_specular_workflow:
            if material.specular_texture is None:
                if material.specular_color is not None:
                    spec_albedo[mask] = material.specular_color.unsqueeze(0)
            else:
                if _texcoords.shape[0] > 0:
                    pixel_val = nvdiffrast.torch.texture(
                        material.specular_texture.unsqueeze(0),
                        _texcoords.reshape(1, 1, -1, 2).contiguous(),
                        filter_mode='linear'
                    )
                    spec_albedo[mask] = pixel_val[0, 0]
        else:
            if material.metallic_texture is None:
                if material.metallic_value is not None:
                    spec_albedo[mask] = (1. - material.metallic_value) * 0.04 + \
                                        albedo[mask] * material.metallic_value
                    albedo[mask] *= (1 - material.metallic_value)
            else:
                if _texcoords.shape[0] > 0:
                    pixel_val = nvdiffrast.torch.texture(
                        material.metallic_texture.unsqueeze(0),
                        _texcoords.reshape(1, 1, -1, 2).contiguous(),
                        filter_mode='nearest'
                    )
                    spec_albedo[mask] = (1. - pixel_val[0, 0]) * 0.04 + albedo[mask] * pixel_val[0, 0]
                    albedo[mask] = albedo[mask] * (1. - pixel_val[0, 0])
        if material.roughness_texture is None:
            if material.roughness_value is not None:
                im_roughness[mask] = torch.clamp(material.roughness_value.unsqueeze(0), 1e-3)
        else:
            if _texcoords.shape[0] > 0:
                pixel_val = nvdiffrast.torch.texture(
                    material.roughness_texture.unsqueeze(0),
                    _texcoords.reshape(1, 1, -1, 2).contiguous(),
                    filter_mode='linear'
                )
                im_roughness[mask] = torch.clamp(pixel_val[0, 0], 1e-3)
            
    img = torch.zeros((1, height, width, 3),
                      dtype=torch.float, device='cuda')
    sg_x, sg_y, sg_z = kal.ops.coords.spherical2cartesian(
        azimuth, elevation)
    directions = torch.stack(
        [sg_y, sg_z, sg_x],
        dim=-1
    )
    
    _im_world_normals = torch.nn.functional.normalize(
        im_world_normals[hard_mask.squeeze(-1)], dim=-1)
    diffuse_effect = kal.render.lighting.sg_diffuse_inner_product(
        amplitude, directions, sharpness,
        _im_world_normals,
        albedo[hard_mask.squeeze(-1)]
    )
    img[hard_mask.squeeze(-1)] = diffuse_effect
    diffuse_img = torch.zeros_like(img)
    diffuse_img[hard_mask.squeeze(-1)] = diffuse_effect

    rays_d = generate_pinhole_rays_dir(camera, height, width)
    specular_effect = kal.render.lighting.sg_warp_specular_term(
        amplitude, directions, sharpness,
        _im_world_normals,
        im_roughness[hard_mask.squeeze(-1)].squeeze(-1),
        -rays_d[hard_mask.squeeze(-1)],
        spec_albedo[hard_mask.squeeze(-1)]
    )
    img[hard_mask.squeeze(-1)] += specular_effect
    specular_img = torch.zeros_like(img)
    specular_img[hard_mask.squeeze(-1)] = specular_effect

    return {
        'img': (torch.clamp(img[0], 0., 1.) * 255.).to(torch.uint8),
    }

def get_random_camera_batch(batch_size, fovy = 0.8575560450553894, iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=1.2, device="cuda"):
    camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(
        *kal.ops.random.sample_spherical_coords((batch_size,), azimuth_low=0., azimuth_high=math.pi * 2,
                                                elevation_low=-math.pi / 2., elevation_high=math.pi / 2., device='cuda'),
        cam_radius
    ), dim=-1)
    return kal.render.camera.Camera.from_args(
        eye=camera_pos + torch.rand((batch_size, 1), device='cuda') * 0.5 - 0.25,
        at=torch.zeros(batch_size, 3),
        up=torch.tensor([[0., 1., 0.]]),
        fov=fovy,
        near=cam_near_far[0], far=cam_near_far[1],
        height=iter_res[0], width=iter_res[1],
        device='cuda'
    )

# # path = os.path.join(COMMON_DATA_DIR, 'meshes', 'gltf_avocado', 'Avocado.gltf')
# objaverse_path = '~/.objaverse/hf-objaverse-v1/glbs/'
# obj_path_list = glob.glob('/home/dongshaocong.vendor/.objaverse/hf-objaverse-v1/glbs/*/*.glb')
# obj_id_list = [p.split('.')[-2].split('/')[-1] for p in obj_path_list]
# # print(obj_id_list[100:150])
# # ff78cedd87a34de49e9633233dffb3d5
# # fe75194122fe42b4bf412bf2dc53eebf
# print('ff78cedd87a34de49e9633233dffb3d5' in obj_id_list)
# print('fe75194122fe42b4bf412bf2dc53eebf' in obj_id_list)
# print("obj_path_list: ", len(obj_path_list))

# debug_idx = obj_id_list.index('3ff33f5baae24b9eab100442fce797e7')
# print("========DEBUG========", obj_path_list[debug_idx])
# # path = 'debug/dress.glb'
# path = obj_path_list[100]
# # path = '/nvme/lihe/workspace/mvdiff_diffusers/examples/lift3d/debug/fe75194122fe42b4bf412bf2dc53eebf.obj'
# print("obj_path: ", path)
# mesh = kal.io.gltf.import_mesh(path)
# # mesh = kal.io.obj.import_mesh(path).cuda()

# mesh = mesh.cuda()
# mesh.materials = [mat.cuda().hwc().contiguous() for mat in mesh.materials]

# # mesh.vertices = kal.ops.pointcloud.center_points(
# #     mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

# vertices = mesh.vertices
# vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
# scale = 0.7 / torch.max(vmax - vmin).item()
# vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
# mesh.vertices = vertices * scale # Rescale to [-0.9, 0.9]

# matrix = np.eye(4)
# matrix[1,1] = 0
# matrix[1,2] = 1
# matrix[2,2] = 0
# matrix[2,1] = -1
# matrix = torch.from_numpy(matrix).to(vertices.device).float()
# verts_pad = torch.cat([mesh.vertices, torch.ones(mesh.vertices.shape[0], 1, device=vertices.device)], dim=1)
# print("===verts pad====", verts_pad.shape) # N, 4
# verts_rot = torch.matmul(verts_pad, matrix)[:, :3]
# mesh.vertices = verts_rot

# azimuth = torch.zeros((1,), device='cuda')
azimuth = torch.zeros((1,), device='cuda')
elevation = torch.full((1,), math.pi / 3., device='cuda')
amplitude = torch.full((1, 3), 3., device='cuda')
# amplitude = torch.full((1, 3), 10., device='cuda')
sharpness = torch.full((1,), 5., device='cuda')

# rt_path = os.path.join('/nvme/lihe/dataset/objaverse_dep_256/00a1a602456f4eb188b522d7ef19e81b/transforms.json')
# with open(rt_path, 'r') as f:
#     transforms = json.load(f)

# c2ws = [frame['transform_matrix'] for frame in transforms['frames']] # Twc
# w2cs = [np.linalg.inv(frame['transform_matrix']) for frame in transforms['frames']] # Tcw
# # Pw = Twc * Pc -> Pc = Tcw * Pw
# c2ws = np.array(c2ws).astype(np.float32)
# w2cs = np.array(w2cs).astype(np.float32) # [nv, 4, 4]

# camera = kal.render.camera.Camera.from_args(view_matrix=w2cs, 
#                                             fov=0.8575560450553894, # 30 * np.pi / 180,
#                                             width=512, height=512,
#                                             device='cuda')

# camera = get_random_camera_batch(1)

# camera = kal.render.camera.Camera.from_args(
#     eye=torch.ones((3,), dtype=torch.float, device='cuda'),
#     at=torch.zeros((3,), dtype=torch.float, device='cuda'),
#     up=torch.tensor([0., 1., 0.], dtype=torch.float),
#     fov=math.pi * 45 / 180,
#     height=512, width=512,
#     near=0.1, far=10000.,
#     device='cuda'
# )
# from diffusers.models.get3d.uni_rep import flex_render
# target = flex_render.render_mesh(mesh, camera, [512, 512], return_types = ["normals"])
# render_normal = (target['normals'] + 1) / 2.
# render_normal = render_normal.permute(0, 3, 1, 2)
# save_image(render_normal, 'debug/test_normals.png', nrow=4)
# exit()

def render(input_camera, input_mesh):
    """Render using camera dimension.
    
    This is the main function provided to the interactive visualizer
    """
    output = base_render(input_camera, input_camera.height, input_camera.width, input_mesh,
                         azimuth, elevation, amplitude, sharpness)
    return output
    
def lowres_render(camera):
    """Render with lower dimension.
    
    This function will be used as a "fast" rendering used when the mouse is moving to avoid slow down.
    """
    output = base_render(camera, int(camera.height / 4), int(camera.width / 4), mesh,
                         azimuth, elevation, amplitude, sharpness)
    return output

# output = render(camera)
# print("==========", output['img'].shape)
# print(output.keys())
# img = output['img']
# img = img.permute(2,0,1).unsqueeze(0) / 255.
# print(img.shape, img.sum())
# save_image(img, 'debug/test_render.png')
# plt.figure()
# plt.imshow(output['img'].cpu().numpy())