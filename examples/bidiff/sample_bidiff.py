import argparse
import os
from pathlib import Path
import gc
import json

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.utils import set_seed
import torchvision
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers import T5EncoderModel

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.bidiff_utils import (
    import_model_class_from_model_name_or_path, 
    model_has_vae, 
    tokenize_prompt, 
    encode_prompt,
    save_model_card,
    get_main_view_tensor,
    )
from diffusers.utils.mv_dataset import BidiffDataset, collate_fn
from diffusers.models.bidiff import BidiffModel
from diffusers.pipelines.bidiff.pipeline_bidiff import BidiffPipeline
from PIL import Image

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
# import rembg


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Bidiff training script.")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        required=True,
        help="Path to checkpoints.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained bidiff model."
        " If not specified whole net weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bidiff-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Text prompts."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["dpm_solver", "uni", "ddpm"],
        help=(
            "Use extra scheduler, now support dpm_solver, uni."
        ),
    )
    parser.add_argument(
        "--sample_stage2",
        action="store_true",
        help=(
            "Sample stage 2."
        ),
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=25,
        help=(
            "The number of inference steps"
        ),
    )
    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help=(
            "Sample in interactive mode."
        ),
    )
    parser.add_argument(
        "--sample_config_file",
        type=str,
        default=None,
        help=(
            "Sample config file for interactive mode."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def get_alpha_inter_ratio(start, end, step):
    if end == 0.0:
        return 1.0
    elif step < start:
        return 0.0
    else:
        return np.min([1.0, (step - start) / (end - start)])

@torch.no_grad()
def main(args):
    if args.seed is None:
        set_seed(args.seed)
    cfg = OmegaConf.load(args.cfg)

    model_id = "DeepFloyd/IF-I-XL-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            local_files_only=True,
        )
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=args.revision, local_files_only=True,
    )
    print('Loading model...')
    if args.unet_model_name_or_path is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_model_name_or_path,
        ).to("cuda")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", revision=args.revision, local_files_only=True,
        ).to("cuda")
    vae = None
    bidiff = BidiffModel.from_pretrained(args.ckpt_path, low_cpu_mem_usage=False, local_files_only=True)
    print('Loading pipeline...')
    weight_dtype = torch.float32
    
    dataset_cfg = cfg.data.train_set
    
    def compute_text_embeddings(prompt):
        with torch.no_grad():
            text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=77)
            prompt_embeds = encode_prompt(
                text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=True,
            )

        return prompt_embeds
    validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

    pre_tokenize = dataset_cfg.get('pre_tokenize', False)
    pre_prompt_encode = dataset_cfg.get('pre_prompt_encode', False)
    prompt_process_args = {}
    if pre_tokenize:
        prompt_process_args['tokenizer'] = tokenizer
        if pre_prompt_encode:
            prompt_process_args['text_encoder'] = text_encoder
    # dataset_cfg.pre_tokenize = False
    # dataset_cfg.pre_prompt_encode = False
    # dataset_cfg.use_pre_process_prompt = False

    dataset_for_pre_process = BidiffDataset(**dataset_cfg, **prompt_process_args)
    dataloader = torch.utils.data.DataLoader(
        dataset_for_pre_process,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )

    def get_text_embed_with_process(prompt):
        if dataset_cfg.use_view_prompt:
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
            # prompts = ['A chair on a black background. ' + args.prompt]
            prompts = [prompt]

        if dataset_cfg.dataset_name in ['shapenet_chair']:
            prompts = ['A chair on a black background. ' + p for p in prompts]
        elif dataset_cfg.dataset_name in ['objaverse_40k', 'objaverse_32k', 'objaverse_42k']:
            prompts = [p + ' Black background.' for p in prompts]
        else:
            raise NotImplementedError

        if dataset_cfg.use_view_prompt:
            prompt_embeds = compute_text_embeddings(prompts).unsqueeze(0)
        else:
            prompt_embeds = compute_text_embeddings(prompts).repeat(8, 1, 1).unsqueeze(0)
        return prompt_embeds, prompts

    bidiff.eval()
    unet.eval()
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)
    pipeline = BidiffPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        bidiff=bidiff,
        scheduler=noise_scheduler,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type
    
    # NOTE(lihe): choose extra scheduler
    if args.scheduler == 'uni':
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        print("========= Using UniPCMultistepScheduler ============")
    elif args.scheduler == 'dpm_solver':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        print("========= Using DPMSolverMultistepScheduler ============")
    else:
        print("========= Using DDPM scheduler ============")

    pipeline = pipeline.to("cuda")

    
    # if args.super_resolution:
    #     sr_model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64", local_files_only=True)
    #     processor = Swin2SRImageProcessor()
    #     rmbg = rembg.new_session(model_name='u2net')
    
    generator = None if args.seed is None else torch.Generator(device=pipeline.device).manual_seed(args.seed)

    for i, batch in enumerate(dataloader):
        if i == 0:
            break
    anneal_start_lod0=0 # NOTE(lihe): hard code for now
    anneal_end_lod0=1000
    alpha_inter_ratio_lod0 = get_alpha_inter_ratio(start=anneal_start_lod0, end=anneal_end_lod0, step=100000)
    mv_images = batch['mv_images'].to(dtype=weight_dtype, device='cuda') # (b,v,c,*sp)
    mv_depths = batch['mv_depths'].to(dtype=weight_dtype, device='cuda') # (b,v,c,*sp)
    b, v, c0, h, w = mv_images.shape
    c2ws = batch['c2ws']
    w2cs = batch['w2cs']
    K = batch['K']
    num_ray_per_view = batch['rays_d'].shape[-2]
    rays_d = batch['rays_d'].reshape([b, -1, 3]).to(dtype=weight_dtype) # (b, v, w*h, 3) -> (b, v*w*h, 3)
    # rays_o = batch['rays_o'].reshape([b, -1, 3])
    rays_o = batch['rays_o'].unsqueeze(2).repeat(1, 1, num_ray_per_view, 1).reshape([b, -1, 3]).to(dtype=weight_dtype) # (b, v, 3) -> (b, v*w*h, 3)
    near_fars = batch['near_fars'].to(dtype=weight_dtype) # (b, 2)
    affine_mat = batch['affine_mat'] # (b, 8, 4, 4)
    # extra views (if any)
    extra_mv_images = batch['extra_mv_images'].to(dtype=weight_dtype) if 'extra_mv_images' in batch.keys() else None# (b,v,c,*sp)
    extra_mv_depths = batch['extra_mv_depths'].to(dtype=weight_dtype) if 'extra_mv_depths' in batch.keys() else None# (b,v,c,*sp)
    extra_c2ws = batch.get('extra_c2ws', None)
    extra_w2cs = batch.get('extra_w2cs', None)
    extra_K = batch.get('extra_K', None)
    extra_rays_d = batch['extra_rays_d'].reshape([b, -1, 3]).to(dtype=weight_dtype) if 'extra_rays_d' in batch.keys() else None # (b, v, w*h, 3) -> (b, v*w*h, 3)
    extra_rays_o = batch['extra_rays_o'].unsqueeze(2).repeat(1, 1, num_ray_per_view, 1).reshape([b, -1, 3]).to(dtype=weight_dtype) if 'extra_rays_o' in batch.keys() else None # (b, v, w*h, 3) -> (b, v*w*h, 3)
    extra_near_fars = batch['extra_near_fars'].to(dtype=weight_dtype) if 'extra_near_fars' in batch.keys() else None # (b, 2)
    extra_affine_mat = batch.get('extra_affine_mat', None)
    sample_rays=dict(rays_d=rays_d, rays_o=rays_o, near_fars=near_fars, K=K, c2ws=c2ws, w2cs=w2cs, affine_mat=affine_mat,
                        mv_images=mv_images, mv_depths=mv_depths,
                        extra_rays_d=extra_rays_d, extra_rays_o=extra_rays_o, extra_near_fars=extra_near_fars, extra_K=extra_K, extra_c2ws=extra_c2ws, extra_w2cs=extra_w2cs, extra_affine_mat=extra_affine_mat,
                        extra_mv_images=extra_mv_images, extra_mv_depths=extra_mv_depths,
                        alpha_inter_ratio_lod0=alpha_inter_ratio_lod0, step=100000,
                        )
    sample_rays['noise_scheduler'] = noise_scheduler
    sample_rays['clean_images'] = mv_images
    voxels = batch.get('voxels', None)
    if voxels is not None:
        voxels = [voxel.to(dtype=weight_dtype, device='cuda') for voxel in voxels]
    sample_rays['voxels'] = voxels

    validation_prompt_negative_prompt_embeds = validation_prompt_negative_prompt_embeds.unsqueeze(1).repeat(b, v, 1, 1)

    for k, value in sample_rays.items():
        if isinstance(value, torch.Tensor):
            sample_rays[k] = value.to(pipeline.device)
    
    while True:

        if args.prompt is None and not args.interactive_mode:
            prompt = batch['caption'][0]
            prompt_embeds = batch["embed"].to(dtype=weight_dtype)
            if prompt_embeds.shape[1] == 1:
                prompt_embeds = prompt_embeds.repeat(1, v, 1, 1)
            kwargs_list = [{
                'prompt': prompt,
                'negative_prompt': "",
                'prompts': [prompt],
                'prompt_embeds': prompt_embeds,
                'negative_prompt_embeds': validation_prompt_negative_prompt_embeds,
                'num_inference_steps': args.num_steps,
            }]
        elif not args.interactive_mode:
            prompt_embeds, prompts = get_text_embed_with_process(args.prompt)
            kwargs_list = [{
                'prompt': args.prompt,
                'negative_prompt': "",
                'prompts': prompts,
                'prompt_embeds': prompt_embeds,
                'negative_prompt_embeds': validation_prompt_negative_prompt_embeds,
                'num_inference_steps': args.num_steps,
            }]
        else:
            command = input(">>>")
            if command == 'q':
                break
            elif command == 'c':
                with open(args.sample_config_file, 'r') as f:
                    sample_cfg = json.load(f)

                kwargs_list = []                
                for j, p in enumerate(sample_cfg['prompts']):
                    prompt_embeds, prompts = get_text_embed_with_process(p)
                    if sample_cfg.get('voxel_path', None) is not None:
                        voxels = [torch.tensor(
                            np.load(sample_cfg['voxel_path'][j]).astype(np.float32), 
                            dtype=weight_dtype, device=prompt_embeds.device)]
                    else:
                        voxels = None
                    for nega_prompt in sample_cfg['negative_prompts']:
                        negative_prompt_embeds = compute_text_embeddings(nega_prompt).unsqueeze(1).repeat(b, v, 1, 1)
                        for i, cgi in enumerate(sample_cfg['cond_guidance_interval']):
                            ctgi = sample_cfg['control_guidance_interval'][i]
                            num_inference_steps = sample_cfg['num_inference_steps'][i]
                            cond_decouple = sample_cfg['cond_decouple'][i]
                            cgs = sample_cfg['cond_guidance_scale'][i]
                            tgs = sample_cfg['text_guidance_scale'][i]                                
                            sample_kwargs = {
                                'prompt': p,
                                'negative_prompt': nega_prompt,
                                'prompts': prompts,
                                'prompt_embeds': prompt_embeds,
                                'negative_prompt_embeds': negative_prompt_embeds,
                                'num_inference_steps': num_inference_steps,
                                'cond_guidance_interval': cgi,
                                'control_guidance_interval': ctgi,
                                'cond_decouple': cond_decouple,
                                'cond_guidance_scale': cgs,
                                'text_guidance_scale': tgs,
                                'voxels': voxels,
                            }
                            kwargs_list.append(sample_kwargs)
            else:
                prompt_embeds, prompts = get_text_embed_with_process(command)
                kwargs_list = [{
                    'prompt': command,
                    'negative_prompt': "",
                    'prompts': prompts,
                    'prompt_embeds': prompt_embeds,
                    'negative_prompt_embeds': validation_prompt_negative_prompt_embeds,
                    'num_inference_steps': args.num_steps,
                    'voxels': voxels,
                }]
        

        print('Sampling...')
        for i, sample_kwargs in enumerate(kwargs_list):
            dir_path = args.output_dir
            os.makedirs(dir_path, exist_ok=True)
            prompt_save = 'sample'
            i = 1
            path = os.path.join(dir_path, '{}_{}.png'.format(prompt_save, i))
            json_save_path = os.path.join(dir_path, '{}_{}.json'.format(prompt_save, i))
            mesh_save_path = os.path.join(dir_path, '{}_{}.ply'.format(prompt_save, i))
            while os.path.exists(path):
                i += 1
                path = os.path.join(dir_path, '{}_{}.png'.format(prompt_save, i))
                json_save_path = os.path.join(dir_path, '{}_{}.json'.format(prompt_save, i))
                mesh_save_path = os.path.join(dir_path, '{}_{}.ply'.format(prompt_save, i))

            sample_rays_input = sample_rays.copy()
            sample_kwargs_save = sample_kwargs.copy()
            sample_kwargs_save.pop('prompts')
            sample_kwargs_save.pop('prompt_embeds')
            sample_kwargs_save.pop('negative_prompt_embeds')
            sample_kwargs_save.pop('voxels')
            sample_kwargs_save['ckpt_path'] = args.ckpt_path
            if sample_kwargs.get('voxels', None) is not None:
                sample_rays_input['voxels'] = sample_kwargs.pop('voxels')
            else:
                sample_kwargs.pop('voxels')
            text = sample_kwargs.pop('prompt')
            prompts = sample_kwargs.pop('prompts')
            print('Prompt: ', prompts[0])
            sample_kwargs.pop('negative_prompt')
            with torch.autocast("cuda", enabled=weight_dtype!=torch.float32):
                images = pipeline(height=64, width=64, # prompt=prompt, 
                                generator=generator, sample_rays=sample_rays_input, unet=unet,
                                weight_dtype=weight_dtype, num_views=8, output_type='pt',
                                noise_scheduler=noise_scheduler,
                                prompt_3d=prompts, 
                                mesh_save_path=mesh_save_path,
                                save_vis=False,
                                **sample_kwargs
                                ).images

            # bv, *chw = images.shape
            # all_images = images.view(b * v, *chw)
            if args.prompt is None and not args.interactive_mode:
                images_save = torch.cat((images, (mv_images.view(b * v, c0, h, w) + 1) / 2.), dim=0)
            else:
                images_save = images
            torchvision.utils.save_image(images_save, path, nrow=4)
            with open(json_save_path, 'w') as f:
                json.dump(sample_kwargs_save, f, indent=4)
            # torchvision.utils.save_image(sample_rays['clean_images'][0], os.path.join(dir_path, 'gt_clean.png'), nrow=4)
            
            # if args.super_resolution:
            #     pixel_values = processor(images_save, return_tensors="pt").pixel_values
            #     _, c, h, w = pixel_values.shape
            #     with torch.no_grad():
            #         outputs = sr_model(pixel_values)
            #         results = outputs.reconstruction.data
            #         outputs_h2 = sr_model(results)
            #         results_h2 = outputs_h2.reconstruction.data
        
        if not args.interactive_mode:
            break

if __name__ == '__main__':
    args = parse_args()
    main(args)