import os
from PIL import Image
from huggingface_hub import create_repo, model_info, upload_folder
import torch
from diffusers import (
    AutoencoderKL
)
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, local_files_only: bool = False):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        local_files_only=local_files_only,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, len(images) // 4, 4).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def model_has_vae(args):
    config_file_name = os.path.join("vae", AutoencoderKL.config_name)
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path, config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path, revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length
    token_data = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return token_data

# def encode_prompt(text_encoder, token_data, text_encoder_use_attention_mask=None):
#     text_input_ids = token_data.input_ids.to(text_encoder.device)
#     if text_encoder_use_attention_mask:
#         attention_mask = token_data.attention_mask.to(text_encoder.device)
#     else:
#         attention_mask = None
#     prompt_embeds = text_encoder(
#         text_input_ids,
#         attention_mask=attention_mask,
#     )
#     prompt_embeds = prompt_embeds[0]
#     return prompt_embeds

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def get_main_view_tensor(x, num_views, main_view_id, has_v_dim=False) -> torch.Tensor:
    """
    Get the main view tensor from the batch.
    :param x: a [(N x Num_view) x C x ...] tensor
    :param num_views: int
    :param main_view_id: an [N] or [N x Num_main] tensor
    :return: a [(N x Num_main) x C x ...] tensor
    """
    if has_v_dim:
        b, num_views, *csp = x.shape
        x_trans = x
    else:
        b = main_view_id.shape[0]
        bv, *csp = x.shape
        x_trans = x.view(b, num_views, *csp)
    b_idx = torch.arange(b).to(x.device)
    if len(main_view_id.shape) == 2:
        num_mv = main_view_id.shape[1]
        if main_view_id.shape[1] == num_views:
            x_trans = x_trans.view(b * num_views, *csp) # NOTE(lihe): bug here
        else:
            v_idx = main_view_id.transpose(0, 1)
            x_trans = x_trans[b_idx, v_idx] # (num_mv, b, *csp)
            x_trans = x_trans.transpose(1, 0).reshape(b * num_mv, *csp)
    else:
        num_mv = 1
        x_trans = x_trans[b_idx, main_view_id] # (b, *csp)
    if has_v_dim:
        x_trans = x_trans.view(b, num_mv, *csp)
    return x_trans