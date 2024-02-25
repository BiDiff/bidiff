import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
import numpy as np
import tqdm
import os

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.models.shap_e.shap_e.models.query import Query
from diffusers.models.shap_e.shap_e.util.collections import AttrDict

from typing import Any, Callable, Dict, Optional
#NOTE(lihe): fix the shap-e ddpm sampling bug 
def uncond_guide_model_x0(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2] # [1, C]
        C = half.shape[1]
        # print("====C====", C)
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs) # [2, C] -> [2, 2C]

        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :C], model_out[:, C:] # [2, C], [2, C]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0) # [2, C] -> [1, C], [1, C]
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps) # [1, C]
        # eps = torch.cat([half_eps, half_eps], dim=0) # [2, C]
        # return torch.cat([eps, rest], dim=1) # [2, 2C]
        return half_eps # [2, C]

    return model_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
# model = load_model('text300M', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))


batch_size = 4
guidance_scale = 3.0


from shap_e.util.image_util import load_image



### dpm solver ###
# scheduler_args = {}
# scheduler_args["variance_type"] = "fixed_small"

# num_inference_steps = 64
# noise_scheduler = DDPMScheduler(num_train_timesteps=1024, beta_schedule='exp', prediction_type="sample")
# # noise_scheduler = DDPMScheduler(num_train_timesteps=1024, beta_schedule='exp')
# # dpm_solver_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config, **scheduler_args)
# dpm_solver_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config, **scheduler_args)
# dpm_solver_scheduler.set_timesteps(num_inference_steps, device=device)
# timesteps = dpm_solver_scheduler.timesteps
# print("====timesteps====", timesteps)
# num_warmup_steps = len(timesteps) - num_inference_steps * dpm_solver_scheduler.order

# latents = torch.randn(batch_size, 1024*1024).to(device)

# # model_kwargs_3d = dict(texts=[prompt] * batch_size)
# model_kwargs_3d = dict(images=[image] * batch_size)
# if hasattr(model, "cached_model_kwargs"):
#     model_kwargs_3d = model.cached_model_kwargs(batch_size, model_kwargs_3d)

# if guidance_scale != 1.0 and guidance_scale != 0.0:
#     for k, v in model_kwargs_3d.copy().items():
#         model_kwargs_3d[k] = torch.cat([v, torch.zeros_like(v)], dim=0)
# # latents = latents * scheduler.init_noise_sigma # TODO(lihe): figure it out
# with torch.no_grad():
#     for i, t in tqdm.tqdm(enumerate(timesteps)):

#         if guidance_scale > 1:
#             latents_input = torch.cat([latents]*2)
#             t_batch = t.repeat(batch_size* 2).view(-1)
#         else:
#             latents_input = latents
#             t_batch = t.repeat(batch_size).view(-1)

#         scaled_model_input = dpm_solver_scheduler.scale_model_input(latents_input, t)
#         if guidance_scale != 1.0:
#             g_model = uncond_guide_model_x0(model, guidance_scale)
#         else:
#             g_model = model
#         output = g_model(scaled_model_input, t_batch, **model_kwargs_3d)
#         output = output.clamp(-1, 1)
#         model_input = dpm_solver_scheduler.step(
#                         output, t, latents, return_dict=False
#                     )[0]
#         latents = model_input
### end dpm solver ###


# TODO(lihe): the karras sampling indeed improve the performance, figure it out later.
# latents = sample_latents(
#     batch_size=batch_size,
#     model=model,
#     diffusion=diffusion,
#     guidance_scale=guidance_scale,
#     # model_kwargs=dict(texts=[prompt] * batch_size),
#     model_kwargs=dict(images=[image] * batch_size),
#     progress=True,
#     clip_denoised=True,
#     # use_fp16=True,
#     use_fp16=False,
#     use_karras=True,
#     # use_karras=False,
#     karras_steps=64,
#     # karras_steps=64,
#     sigma_min=1e-3,
#     sigma_max=160,
#     s_churn=0,
# )

# Example of saving the latents as meshes.
# from shap_e.util.notebooks import decode_latent_mesh

# for i, latent in enumerate(latents):
#     t = decode_latent_mesh(xm, latent).tri_mesh()
#     with open(f'debug_data/example_mesh_{i}.ply', 'wb') as f:
#         t.write_ply(f)
#     with open(f'debug_data/example_mesh_{i}.obj', 'w') as f:
#         t.write_obj(f)

def create_coarse_latents(obj_id, img_path, save_path, query_pts=None, render_img=False, render_mode='nerf', size=64):
    image = load_image(img_path)
    # TODO(lihe): the karras sampling indeed improve the performance, figure it out later.
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        # model_kwargs=dict(texts=[prompt] * batch_size),
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        # use_fp16=True,
        use_fp16=False,
        use_karras=True,
        # use_karras=False,
        karras_steps=64,
        # karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    # NOTE(lihe): debug
    os.makedirs(os.path.join(save_path, obj_id), exist_ok=True)
    save_fp = os.path.join(save_path, obj_id, 'coarse_latents.npy')
    np.save(save_fp, latents.detach().cpu().numpy())
    # debug
    # latents = np.load('/home/lihe/dataset/latents/1ab42ccff0f8235d979516e720d607b8/latent.npy')
    # latents = torch.from_numpy(latents).to(device)
    # end debug
    if render_img:
        cameras = create_pan_cameras(size, device)
        for i, latent in enumerate(latents):
            print("==latent===", latent.shape)
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            # display(gif_widget(images))
            images[0].save('./debug_data/results_coarse_{}.gif'.format(i), save_all=True, append_images=images[1:], duration=100, loop=0)
            # images[0].save('./debug_data/results_sample_ddpm_debug_{}.gif'.format(i), save_all=True, append_images=images[1:], duration=100, loop=0)
            # images[0].save('./debug_data/results_sample_karras_debug_{}.gif'.format(i), save_all=True, append_images=images[1:], duration=100, loop=0)
            # images[0].save('./debug_data/results_final_{}.gif'.format(i), save_all=True, append_images=images[1:], duration=100, loop=0)
            print("======saved {}".format(i))
    
    # NOTE(lihe): rotate
    # x = -y, y = x
    rot_mat = np.array([[0, -1., 0],
                            [1., 0, 0],
                            [0, 0., 1.]]) # 3x3
    rot_mat = torch.from_numpy(rot_mat).to(device).float()
    query_pts = query_pts @ rot_mat 
    if query_pts is not None:
        query = Query(
                position=query_pts / 0.35,
                direction=None,
            )
        options_3d = AttrDict(rendering_mode=render_mode, render_with_direction=False)
        # get params
        with torch.no_grad():
            # print("============ render 3d prior =============")
            params = xm.encoder.bottleneck_to_params(
                latents[0][None]
            )
            raw = xm.renderer._query(query=query, params=params, options=options_3d)
            density = raw.density
            sdf = raw.signed_distance
            color = raw.channels
            save_field = torch.cat([density, sdf, color], dim=-1)
            np.save(os.path.join(save_path, obj_id, 'grid_field.npy'), save_field.detach().cpu().numpy())
            # print("============ end 3d prior =============")

if __name__ == '__main__':
    data_root = '/home/lihe/dataset/tmp_mv_256/img/03001627' # shapnet
    save_path = '/home/lihe/dataset/coarse_data'
    id_list = os.listdir(data_root)
    query_pts = np.load('/home/lihe/workspace/mvdiff_diffusers/examples/lift3d/debug/grid_pts.npy')
    query_pts = torch.from_numpy(query_pts).to(device)
    for obj_id in tqdm.tqdm(id_list):
        # obj_id = '1ab4c6ef68073113cf004563556ddb36'
        obj_id = '1ab42ccff0f8235d979516e720d607b8'
        print("===obj id is : ", obj_id)
        img_path = os.path.join(data_root, obj_id, '005.png')
        create_coarse_latents(obj_id, img_path, save_path, query_pts, render_img=True)
        break


