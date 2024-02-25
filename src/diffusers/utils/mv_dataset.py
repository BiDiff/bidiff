import os
import json
import csv
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import glob
import trimesh

from diffusers import (
    DiffusionPipeline
)

import pandas as pd
import kaolin as kal


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length
    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs


class GeneralMVDataset(Dataset):
    def __init__(
        self,
        root,
        instance_list,
        dataset_name=None,
        path_prefix='',
        path_infix='',
        path_suffix='png',
        file_suffix='',
        transform=None,
        size=None,
        rays_res=None,
        use_depth=False,
        use_de_depth=False,
        depth_estimation_pipline=None,
        use_pose=False,
        path_prefix_dep='',
        path_infix_dep='',
        path_suffix_dep='png0001.png',
        file_suffix_dep='',
        transform_dep=None,
        load_views=8,
        num_main_views=8,
        is_train=True,
        use_cache=True,
        extra_view_root=None,
        extra_load_views=8,
        use_3d_prior=False,
        sdf_gen=False,
        voxel_cond=False,
        load_obj=False,
        refer_view_id=None,
        render_img_online=False,
        latent_root=None,
        sdf_root=None,
        voxel_root=None,
        obj_root=None,
        mesh_npy_root=None,
        get_res_64=False,
        **kwargs
    ):
        super().__init__()
        self.root = root
        self.extra_root = extra_view_root
        self.instance_list = instance_list
        self.dataset_name = dataset_name
        self.path_prefix = path_prefix
        self.path_infix = path_infix
        self.path_suffix = path_suffix
        self.file_suffix = file_suffix
        self.transform = transform
        self.size = size
        if rays_res is not None:
            self.rays_res = rays_res
        else:
            self.rays_res = size
        self.use_depth = use_depth
        self.use_de_depth = use_de_depth
        self.depth_estimation_pipline = depth_estimation_pipline
        self.use_pose = use_pose
        self.path_prefix_dep = path_prefix_dep
        self.path_infix_dep = path_infix_dep
        self.path_suffix_dep = path_suffix_dep
        self.file_suffix_dep = file_suffix_dep
        self.transform_dep = transform_dep
        if not isinstance(load_views, int):
            load_views = list(load_views)
        if not isinstance(extra_load_views, int):
            extra_load_views = list(extra_load_views)
        self.load_views = load_views
        self.extra_load_views = extra_load_views
        self.num_main_views = num_main_views
        self.is_train = is_train
        self.use_cache = use_cache
        self.use_3d_prior = use_3d_prior
        self.sdf_gen = sdf_gen
        self.voxel_cond = voxel_cond
        self.load_obj = load_obj
        self.render_img_online = render_img_online
        self.latent_root = latent_root
        self.sdf_root = sdf_root # TODO(lihe): specify the sdf path
        self.voxel_root = voxel_root # TODO(lihe): specify the sdf path
        self.obj_root = obj_root
        self.refer_view_id = refer_view_id
        self.get_res_64 = get_res_64

        if self.load_obj:
            self.obj_path_list = glob.glob(os.path.join(self.obj_root, '*/*.glb'))
            self.obj_id_list = [p.split('.')[-2].split('/')[-1] for p in self.obj_path_list]
            # save faces and vertices for every mesh
            print("=========saving vertices and faces from raw glb files==========")
            self.mesh_npy_root = mesh_npy_root
            invalid_count = 0
            for instance_name in tqdm(self.instance_list, total=len(self.instance_list)):
                if instance_name not in self.obj_id_list:
                    continue
                idx = self.obj_id_list.index(instance_name)
                obj_path = self.obj_path_list[idx]
                save_path = os.path.join(self.mesh_npy_root, instance_name + '.npz')
                if not os.path.exists(save_path):
                    try:
                        mesh_data = trimesh.load(obj_path, force='mesh') # NOTE(lihe): use mesh data to load gt mesh
                        mesh_v = np.array(mesh_data.vertices)
                        mesh_v = torch.from_numpy(mesh_v).float()
                        mesh_f = np.array(mesh_data.faces)
                        mesh_f = torch.from_numpy(mesh_f)
                        np.savez(save_path, mesh_v=mesh_v.numpy(), mesh_f=mesh_f.numpy())
                    except:
                        invalid_count += 1
            print("=========invalid obj num: {}==========".format(invalid_count))
            self.mesh_npy_path_list = glob.glob(os.path.join(self.mesh_npy_root, '*.npz'))
            self.mesh_npy_id_list = [p.split('/')[-1].split('.')[0] for p in self.mesh_npy_path_list]

        # filtering instances
        if self.extra_root is not None:
            valid_list = []
            for ins_id in tqdm(self.instance_list, total=len(self.instance_list)):
                extra_instance_path = os.path.join(self.extra_root, self.path_prefix, ins_id, self.path_infix)
                image_path = os.path.join(extra_instance_path, '{:03d}{}.{}'.format(7, self.file_suffix, self.path_suffix))
                pose_path = os.path.join(extra_instance_path, 'transforms.json')
                if use_3d_prior:
                    latent_path = os.path.join(self.latent_root, ins_id, 'latent.npy')
                    latent_flag = os.path.exists(latent_path)
                else:
                    latent_flag = True
                if sdf_gen:
                    # sdf_path = os.path.join(self.sdf_root, ins_id, 'grid_gt.csv')
                    sdf_path = os.path.join(self.sdf_root, ins_id + '.npy')
                    sdf_flag = os.path.exists(sdf_path)
                else:
                    sdf_flag = True
                if self.voxel_cond:
                    voxel_path = os.path.join(self.voxel_root, ins_id, 'voxel.npy')
                    voxel_flag = os.path.exists(voxel_path)
                else:
                    voxel_flag = True
                if self.load_obj:
                    # obj_path = os.path.join(self.obj_root, ins_id + '.obj')
                    # obj_flag = os.path.exists(obj_path)
                    obj_flag = (ins_id in self.obj_id_list) and (ins_id in self.mesh_npy_id_list)
                else:
                    obj_flag = True
                if os.path.exists(image_path) and os.path.exists(pose_path) and latent_flag and sdf_flag and voxel_flag and obj_flag:
                    valid_list.append(ins_id)
            print("==========instance num filtered from {} to {}========".format(len(self.instance_list), len(valid_list)))

            
            self.instance_list = valid_list

        if isinstance(self.load_views, int):
            self.num_views = self.load_views
        elif isinstance(self.load_views, list):
            self.num_views = len(self.load_views)
        else:
            raise TypeError('load_views must be int or list')
        
        if self.use_cache:
            self.cache = [None] * self.__len__()
                
        
    def __len__(self):
        return len(self.instance_list)
    
    def get_rays(self, instance_path, res, load_views, num_views=None):
        data = {}
        rt_path = os.path.join(instance_path, 'transforms.json')
        with open(rt_path, 'r') as f:
            transforms = json.load(f)
        if isinstance(load_views, list):
            transforms['frames'] = [transforms['frames'][view] for view in load_views]
        elif isinstance(load_views, int):
            transforms['frames'] = transforms['frames'][:load_views]
        c2ws = [frame['transform_matrix'] for frame in transforms['frames']] # Twc
        w2cs = [np.linalg.inv(frame['transform_matrix']) for frame in transforms['frames']] # Tcw
        # Pw = Twc * Pc -> Pc = Tcw * Pw
        c2ws = np.array(c2ws).astype(np.float32)
        w2cs = np.array(w2cs).astype(np.float32) # [nv, 4, 4]
        data['c2ws'] = c2ws
        data['w2cs'] = w2cs

        # K
        fov = transforms['camera_angle_x'] # 0.8575560450553894
        fx = - res * 35 / 32.
        fy = res * 35 / 32.
        cx = res / 2
        cy = res / 2
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        data['K'] = K.astype(np.float32) # [3, 3]

        # affine mats
        affine_mat_list = []
        for w2c in w2cs:
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = K @ w2c[:3, :4]
            affine_mat_list.append(affine_mat)
        affine_mat = np.stack(affine_mat_list, axis=0) # [nv, 4, 4]
        data['affine_mat'] = affine_mat.astype(np.float32)

        # rays
        x, y = torch.meshgrid(torch.linspace(-1, 1, res),
                            torch.linspace(1, -1, res))  # W,H
        x = x.T.flatten()  # W*H
        y = y.T.flatten()  # W*H
        z = -torch.ones_like(x) / \
            np.tan(fov/2)  # W*H
        rays_cam = torch.stack([x, y, z], dim=-1) # W*H, 3
        # rays_cam = rays_cam * np.tan(fov/2) # normalize to depth [W*H, 3]
        rays_cam_norm = normalize_vecs(rays_cam) # [W*H, 3]
        rays_d_list = []
        rays_o_list = []
        if num_views is None:
            num_views = self.num_views
        for v_id in range(num_views):
            R = c2ws[v_id][:3, :3]
            rays_d = rays_cam_norm @ R.T
            rays_d_list.append(rays_d)
            rays_o = c2ws[v_id][:3, 3]
            rays_o_list.append(rays_o)
        rays_d = np.stack(rays_d_list, axis=0)
        rays_o = np.stack(rays_o_list, axis=0) # [nv, num_rays, 3]
        data['rays_d'] = rays_d
        data['rays_o'] = rays_o
        data['near_fars'] = np.array([0.8, 1.6])

        # pose_embed
        # w2cs_R, w2cs_T = w2cs[:, :3, :3], w2cs[:, :3, 3:4]
        # camera_t_ = (-w2cs_R.transpose(0, 2, 1) @ w2cs_T)[:, :, 0]
        camera_t = c2ws[:, :3, 3]

        theta, azimuth, z = self.cartesian_to_spherical(camera_t)
        azimuth = azimuth % (2 * np.pi)
        if self.refer_view_id is not None:
            theta_ref = theta[self.refer_view_id: self.refer_view_id+1]
            azimuth_ref = azimuth[self.refer_view_id: self.refer_view_id+1]
            z_ref = z[self.refer_view_id: self.refer_view_id+1]
            theta_rel = theta - theta_ref
            azimuth_rel = azimuth - azimuth_ref
            z_rel = z - z_ref
            pos_embed = np.stack([theta_rel, np.sin(azimuth_rel), np.cos(azimuth_rel), z_rel], axis=1)
        else:
            pos_embed = np.stack([theta, np.sin(azimuth), np.cos(azimuth), z], axis=1)
        data['pos_embed'] = pos_embed # [nv, 4]

        return data

    def cartesian_to_spherical(self, xyz):
        # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return theta, azimuth, z
    
    def rand_anoter(self):
        idx = np.random.randint(self.__len__())
        return self.__getitem__(idx)
    
    def __getitem__(self, index):
        if self.use_cache and self.cache[index] is not None:
            return self.cache[index]
        data = {}
        instance_name = self.instance_list[index]
        instance_path = os.path.join(self.root, self.path_prefix, instance_name, self.path_infix)
        if self.use_depth:
            instance_path_dep = os.path.join(
                self.root, self.path_prefix_dep, instance_name, self.path_infix_dep)
            depth_images = []
        if self.use_de_depth:
            de_depth_images = []

        # mv images
        images = []
        images_64 = []
        if isinstance(self.load_views, int):
            for v in range(self.load_views):
                image_path = os.path.join(instance_path, '{:03d}{}.{}'.format(v, self.file_suffix, self.path_suffix))
                image = Image.open(image_path).convert('RGB')
                if self.get_res_64:
                    image_64 = image.resize((64, 64), Image.BILINEAR)
                    image_64 = F.to_tensor(image_64)
                    image_64 = F.normalize(image_64, [0.5], [0.5])
                    images_64.append(image_64)
                if self.use_de_depth:
                    de_depth = self.depth_estimation_pipline(image)["depth"]
                    if self.transform_dep is not None:
                        de_depth = self.transform_dep(de_depth)
                    de_depth_images.append(de_depth)
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
                if self.use_depth:
                    depth_path = os.path.join(instance_path_dep, '{:03d}{}.{}'.format(v, self.file_suffix_dep, self.path_suffix_dep))
                    depth = Image.open(depth_path).convert('RGB')
                    if self.transform_dep is not None:
                        depth = self.transform_dep(depth)
                    depth_images.append(depth)
        elif isinstance(self.load_views, list):
            for v in self.load_views:
                image_path = os.path.join(instance_path, '{:03d}{}.{}'.format(v, self.file_suffix, self.path_suffix))
                image = Image.open(image_path).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
                if self.use_depth:
                    depth_path = os.path.join(instance_path_dep, '{:03d}{}.{}'.format(v, self.file_suffix_dep, self.path_suffix_dep))
                    depth = Image.open(depth_path).convert('RGB')
                    if self.transform_dep is not None:
                        depth = self.transform_dep(depth)
                    depth_images.append(depth)
        
        images = torch.stack(images, dim=0)
        data['mv_images'] = images
        if self.get_res_64:
            images_64 = torch.stack(images_64, dim=0)
            data['mv_images_64'] = images_64
        if self.use_depth:
            depth_images = torch.stack(depth_images, dim=0)
            data['mv_depths'] = depth_images
        if self.use_de_depth:
            de_depth_images = torch.stack(de_depth_images, dim=0)
            data['de_depths'] = de_depth_images
        
        if self.num_main_views == 1:
            data['main_view_id'] = torch.randint(0, self.num_views, ())
        elif self.num_main_views < self.num_views:
            data['main_view_id'] = torch.randperm(self.num_views)[:self.num_main_views]
        else:
            data['main_view_id'] = torch.arange(self.num_views)
        
        # mv RT
        if self.use_pose:
            # res = images.shape[-1]
            data.update(self.get_rays(instance_path, self.rays_res, self.load_views))

        # extra view
        if self.extra_root is not None:
            extra_instance_path = os.path.join(self.extra_root, self.path_prefix, instance_name, self.path_infix)
            extra_images = []
            if self.use_depth:
                extra_instance_path_dep = os.path.join(
                    self.extra_root, self.path_prefix_dep, instance_name, self.path_infix_dep)
                extra_depth_images = []
            # load extra images and depths
            if isinstance(self.extra_load_views, int):
                for v in range(self.extra_load_views):
                    image_path = os.path.join(extra_instance_path, '{:03d}{}.{}'.format(v, self.file_suffix, self.path_suffix))
                    image = Image.open(image_path).convert('RGB')
                    if self.transform is not None:
                        image = self.transform(image)
                    extra_images.append(image)
                    if self.use_depth:
                        depth_path = os.path.join(extra_instance_path_dep, '{:03d}{}.{}'.format(v, self.file_suffix_dep, self.path_suffix_dep))
                        depth = Image.open(depth_path).convert('RGB')
                        if self.transform_dep is not None:
                            depth = self.transform_dep(depth)
                        extra_depth_images.append(depth)
            elif isinstance(self.extra_load_views, list):
                for v in self.extra_load_views:
                    image_path = os.path.join(extra_instance_path, '{:03d}{}.{}'.format(v, self.file_suffix, self.path_suffix))
                    image = Image.open(image_path).convert('RGB')
                    if self.transform is not None:
                        image = self.transform(image)
                    extra_images.append(image)
                    if self.use_depth:
                        depth_path = os.path.join(extra_instance_path_dep, '{:03d}{}.{}'.format(v, self.file_suffix_dep, self.path_suffix_dep))
                        depth = Image.open(depth_path).convert('RGB')
                        if self.transform_dep is not None:
                            depth = self.transform_dep(depth)
                        extra_depth_images.append(depth)
            else:
                raise NotImplementedError

            # shuffle
            if isinstance(self.extra_load_views, int):
                num_extra_load_views = self.extra_load_views
                inds = torch.randperm(num_extra_load_views)
            elif isinstance(self.extra_load_views, list):
                num_extra_load_views = len(self.extra_load_views)
                inds = torch.randperm(num_extra_load_views)
            extra_images = torch.stack(extra_images, dim=0)[inds]
            data['extra_mv_images'] = extra_images
            if self.use_depth:
                data['extra_mv_depths'] = torch.stack(extra_depth_images, dim=0)[inds]
            
            # RT
            extra_rays = self.get_rays(extra_instance_path, self.rays_res, self.extra_load_views)
            for key in extra_rays.keys():
                new_key = 'extra_' + key
                data[new_key] = extra_rays[key][inds] if extra_rays[key].shape[0] == num_extra_load_views else extra_rays[key]
        
        if self.use_3d_prior:
            latent_path = os.path.join(self.latent_root, instance_name, 'latent.npy')
            latent = np.load(latent_path)
            assert len(latent.shape) == 2
            latent = latent[0]
            data['latents'] = latent
        
        if self.sdf_gen:
            sdf_path = os.path.join(self.sdf_root, instance_name + '.npy')
            gt_sdf = np.load(sdf_path)
            data['gt_sdf'] = gt_sdf
        
        if self.voxel_cond:
            voxel_path = os.path.join(self.voxel_root, instance_name, 'voxel.npy')
            voxel = np.load(voxel_path).astype(np.float32)
            data['voxels'] = voxel
        
        if self.load_obj:
            # obj_path = os.path.join(self.obj_root, instance_name + '.obj')
            # gt_mesh = kal.io.obj.import_mesh(obj_path)
            idx = self.obj_id_list.index(instance_name)
            obj_path = self.obj_path_list[idx]
            npy_path = os.path.join(self.mesh_npy_root, instance_name + '.npz')
            mesh_data = np.load(npy_path)
            mesh_v = mesh_data['mesh_v']
            mesh_v = torch.from_numpy(mesh_v).float()
            mesh_f = mesh_data['mesh_f']
            mesh_f = torch.from_numpy(mesh_f)        
            gt_mesh = kal.rep.SurfaceMesh(mesh_v, mesh_f)

            vertices = gt_mesh.vertices
            vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
            scale = 0.7 / torch.max(vmax - vmin).item()
            vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
            gt_mesh.vertices = vertices * scale # Rescale to [-0.9, 0.9]

            # rotate glb mesh
            matrix = np.eye(4)
            matrix[1,1] = 0
            matrix[1,2] = 1
            matrix[2,2] = 0
            matrix[2,1] = -1
            matrix = torch.from_numpy(matrix).to(vertices.device).float()
            verts_pad = torch.cat([gt_mesh.vertices, torch.ones(gt_mesh.vertices.shape[0], 1, device=vertices.device)], dim=1)
            verts_rot = torch.matmul(verts_pad, matrix)[:, :3]
            gt_mesh.vertices = verts_rot
            data['gt_mesh'] = gt_mesh

            # render img online
            if self.render_img_online:
                from diffusers.models.get3d.uni_rep.kaolin_render import render
                mesh = gt_mesh.cuda()
                mesh.materials = [mat.cuda().hwc().contiguous() for mat in mesh.materials]
                w2cs = data['w2cs']
                camera = kal.render.camera.Camera.from_args(view_matrix=w2cs, 
                                            fov=0.8575560450553894, # 30 * np.pi / 180,
                                            width=512, height=512,
                                            device='cuda')
                img_list = []
                for v_id in range(self.num_views):
                    with torch.no_grad():
                        output = render(camera[v_id], input_mesh=mesh)
                    img = output['img']
                    img = img.permute(2,0,1).unsqueeze(0) / 255. # 0~1 # 512,512,3 -> 1,3,512,512
                    img = img * 2 - 1.
                    img_list.append(img)
                mv_images_high = torch.cat(img_list, dim=0) # 8,3,512,512
                data['mv_images_high'] = mv_images_high
                
                del mesh
                del camera

        if self.use_cache:
            assert self.cache[index] is None
            self.cache[index] = data
        return data
    

class GeneralPromptDataset(Dataset):
    def __init__(
        self,
        prompt_file,
        instance_list,
        dataset_name=None,
        use_view_prompt=False,
        pre_tokenize=False,
        tokenizer=None,
        tokenizer_max_length=None,
        use_pre_process_prompt=True,
        pre_prompt_encode=False,
        prompt_encoder=None,
        use_token_attention_mask=False,
        pre_prompt_embed_cache=None,
        **kwargs
    ):
        super().__init__()
        self.prompt_file = prompt_file
        self.instance_list = instance_list
        self.dataset_name = dataset_name
        self.use_view_prompt = use_view_prompt
        self.pre_tokenize=pre_tokenize
        self.tokenizer=tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.use_pre_process_prompt = use_pre_process_prompt
        self.pre_prompt_encode=pre_prompt_encode
        self.prompt_encoder=prompt_encoder
        self.use_token_attention_mask = use_token_attention_mask
        self.pre_prompt_embed_cache = pre_prompt_embed_cache

        if os.path.basename(prompt_file).split('.')[-1] == 'json':
            with open(prompt_file, 'r') as f:
                prompt_dict = json.load(f)
        elif os.path.basename(prompt_file).split('.')[-1] == 'csv':
            if dataset_name in ['shapenet_chair']:
                prompt_list = []
                with open(prompt_file, 'r') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        prompt_list.append(row)
                prompt_dict = {} 
                for p in prompt_list:
                    if p[4] == '03001627' and p[1] not in prompt_dict:
                        prompt_dict[p[1]] = p[2]
            elif dataset_name in ['objaverse_32k']:
                prompt_dict = {}
                with open(prompt_file, 'r') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        prompt_dict[row[0]] = row[1]
            else:
                raise NotImplementedError

        self.prompt_list = []
        # self.prompt_dict = {}
        for ins in tqdm(instance_list, total=len(instance_list)):
            if ins in prompt_dict:
                prompt = prompt_dict[ins]
                if self.use_pre_process_prompt:
                    prompt = self.pre_process_prompt(prompt, ins)
                    self.prompt_list.append(prompt)
                else:
                    self.prompt_list.append({'caption': prompt})
                # self.prompt_dict[ins] = prompt
            else:
                raise ValueError('{} is not in prompt file.'.format(ins))
            
        self.tokenizer = None
        self.prompt_encoder = None

    @torch.no_grad()
    def pre_process_prompt(self, prompt, ins=None) -> dict:
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        if self.use_view_prompt:
            prompts = [
                'Right side view. ' + prompt,
                'Right oblique rear view. ' + prompt,
                'Rear view. ' + prompt,
                'Left oblique rear view. ' + prompt,
                'Left view. ' + prompt,
                'Left oblique front view. ' + prompt,
                'Front view. ' + prompt,
                'Right oblique front view. ' + prompt,
            ] # Hard
        else:
            prompts = [prompt]

        
        # if self.dataset_name in ['shapenet_chair', 'objaverse_40k']:
        if self.dataset_name in ['shapenet_chair']:
            prompts = ['A chair on a black background. ' + p for p in prompts]
        elif self.dataset_name in ['objaverse_40k', 'objaverse_32k', 'objaverse_42k']:
            prompts = [p + ' Black background.' for p in prompts]
        else:
            raise NotImplementedError

        if self.pre_tokenize:
            token = tokenize_prompt(self.tokenizer, prompts, self.tokenizer_max_length)
            token_ids = token.input_ids
            token_attention_mask = token.attention_mask
            if self.pre_prompt_encode:
                if not self.use_token_attention_mask:
                    token_attention_mask = None
                else:
                    token_attention_mask = token_attention_mask.to(self.prompt_encoder.device)
                if self.pre_prompt_embed_cache is not None:
                    embed_path = os.path.join(self.pre_prompt_embed_cache, '{}/prompt_embed.npy'.format(ins))
                    # if True:
                    if not os.path.exists(embed_path):
                        prompt_embeds = self.prompt_encoder(
                            token_ids.to(self.prompt_encoder.device),
                            attention_mask=token_attention_mask,
                        )[0].cpu()
                        prompt_embeds_np = prompt_embeds.float().detach().numpy()
                        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                        np.save(embed_path, prompt_embeds_np)
                    else:
                        prompt_embeds = embed_path
                    
                else:
                    prompt_embeds = self.prompt_encoder(
                        token_ids.to(self.prompt_encoder.device),
                        attention_mask=token_attention_mask,
                    )[0].cpu()
                processed_prompts = {'caption': prompts, 'embed': prompt_embeds}
            else:
                processed_prompts = {
                    'caption': prompts,
                    'token_ids': token_ids, 
                    'token_attention_mask': token_attention_mask,
                }
        else:
            processed_prompts = {'caption': prompts}
        
        return processed_prompts
            
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        data = self.prompt_list[index]
        if 'embed' in data and isinstance(data['embed'], str):
            data['embed'] = torch.from_numpy(np.load(data['embed']))
        return data


class BidiffDataset(Dataset):
    def __init__(
        self,
        root,
        instance_file,
        transform=None,
        use_depth=False,
        transform_dep=None,
        tokenizer=None,
        text_encoder=None,
        **kwargs
    ):
        super().__init__()
        with open(instance_file, 'r') as f:
            instance_list = [fn.strip() for fn in f.readlines()]
        
        if transform is not None:
            image_transforms = [transforms.Resize(transform.size, antialias=True)]
            size = transform.size
            if use_depth:
                depth_transforms = [transforms.ToTensor()]
        else:
            size = None
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
        self.image_transforms = transforms.Compose(image_transforms)
        if use_depth:
            depth_transforms.extend([transforms.Lambda(lambda x: torch.where(x < 0.99, x, torch.zeros_like(x))[0:1]),
                                     transforms.Resize(transform_dep.size, antialias=True),
                                     transforms.Lambda(lambda x: x[0])])
            self.depth_transforms = transforms.Compose(depth_transforms)
        else:
            self.depth_transforms = None

        self.mv_img_dataset = GeneralMVDataset(
            root=root,
            instance_list=instance_list,
            transform=self.image_transforms,
            transform_dep=self.depth_transforms,
            use_depth=use_depth,
            size=size,
            **kwargs
        )

        # check if the instance list has been filtered by MVDataset
        if len(self.mv_img_dataset) < len(instance_list):
            instance_list = self.mv_img_dataset.instance_list
            print("====== Filtered ! =======")

        self.prompt_dataset = GeneralPromptDataset(
            instance_list=instance_list,
            tokenizer=tokenizer,
            prompt_encoder=text_encoder,
            **kwargs
        )

        self.root = root
        self.instance_list = instance_list


    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        data = self.mv_img_dataset.__getitem__(index)
        data.update(self.prompt_dataset.__getitem__(index))
        return data
    
    def dreambooth_generation(self, args, generation_dir, device='cuda', start_id=0, end_id=1000):
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=args.revision,
            local_files_only=True,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(device)
        
        for index, ins in tqdm(enumerate(self.instance_list[start_id:end_id]), total=len(self.instance_list[start_id:end_id])):
            prompt_data = self.prompt_dataset.__getitem__(index)
            ins_dir = os.path.join(generation_dir, ins)
            if not os.path.exists(os.path.join(ins_dir, '007.png')):
                os.makedirs(ins_dir, exist_ok=True)
            else:
                continue
            prompts = prompt_data['caption']
            if len(prompts) == 1:
                prompts = prompts * 8
            assert len(prompts) == 8
            images = pipeline(prompts).images

            for j, image in enumerate(images):
                image_filename = os.path.join(ins_dir, '{:03d}.png'.format(j))
                image.save(image_filename)
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collate_fn(examples):
    stack_tensor_keys = ['mv_images', 'mv_depths', 'de_depths', 'token_ids', 'token_attention_mask', 'embed', 'main_view_id', 
                         'K', 'c2ws', 'w2cs', 'rays_o', 'rays_d', 'affine_mat','near_fars', 'latents', 'gt_sdf', 'pos_embed',
                         'mv_images_high']
    extra_stack_tensor_keys = ['extra_' + x for x in stack_tensor_keys]
    list_tensor_keys = ['voxels'] # []
    other_keys = ['caption', 'gt_mesh'] # , 'caption_ref'
    keys = examples[0].keys()
    data_dict = {}
    for k in keys:
        if k in stack_tensor_keys or k in extra_stack_tensor_keys:
            data_stack = torch.stack([torch.tensor(e[k]) if isinstance(e[k], np.ndarray) else e[k] for e in examples], dim=0)
            data_stack = data_stack.to(memory_format=torch.contiguous_format)
            data_dict[k] = data_stack
        elif k in list_tensor_keys:
            data_dict[k] = [torch.tensor(e[k], dtype=torch.float32).to(
                memory_format=torch.contiguous_format) for e in examples]
        elif k in other_keys:
            data_dict[k] = [e[k] for e in examples]

    return data_dict