import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import gc

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from huggingface_hub import create_repo, model_info, upload_folder
from packaging import version
import torchvision
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
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
from diffusers.models.modeling_utils import load_state_dict

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


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
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bidiff_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained bidiff model."
        " If not specified whole net weights are initialized from unet.",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
        help="Path to pretrained bidiff model that includes controlnet.",
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bidiff-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    # parser.add_argument(
    #     "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    # )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--mv_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--shape_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help=(
            "Number of sample steps."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_lift3d",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--classifier_free_guidance_training_rate",
        type=float,
        default=0.1,
        help="Rate of replacing the prompt as uncond prompt when training if use classifier free guidance",
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Set to not save text encoder"
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=1000,
        help=(
            "only train part timesteps."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the unet encoder."
        )
    
    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args

def get_alpha_inter_ratio(start, end, step):
    if end == 0.0:
        return 1.0
    elif step < start:
        return 0.0
    else:
        return np.min([1.0, (step - start) / (end - start)])


def main(args):
    cfg = OmegaConf.load(args.cfg)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator_init_process_config = InitProcessGroupKwargs(timeout = timedelta(seconds=7200)) # NOTE(lihe): increasing timeout from 1800 to 7200
    distributed_data_parallel_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_init_process_config, distributed_data_parallel_config] # 
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            local_files_only=True,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, local_files_only=True)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, local_files_only=True,
    )

    model_cfg = cfg.model
    vae = None

    if args.unet_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_model_name_or_path,
            local_files_only=True,
        )
        logger.info("Loading local unet weights")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True,
        )
    unet.requires_grad_(False)
    if args.bidiff_model_name_or_path:
        logger.info("Loading existing bidiff weights")
        bidiff = BidiffModel.from_pretrained(args.bidiff_model_name_or_path, low_cpu_mem_usage=False, local_files_only=True) # low_cpu_mem_usage=False for nn.utils.weight_norm
    else:
        logger.info("Initializing bidiff weights from unet")
        if model_cfg.get('view_attn_id', []) != []:
            assert not cfg.data.train_set.use_view_prompt
        device = accelerator.device if model_cfg.use_3d_prior else 'cuda'
        device = 'cuda'
        bidiff = BidiffModel.from_unet(unet, model_cfg, device=device)
    if args.controlnet_model_path:
        pt_path = os.path.join(args.controlnet_model_path, 'diffusion_pytorch_model.bin')
        state_dict = load_state_dict(pt_path)
        ks = list(state_dict.keys())
        for k in ks:
            if 'denoiser3d.' in k:
                state_dict.pop(k)
        bidiff._convert_deprecated_attention_blocks(state_dict)
        bidiff, missing_keys, unexpected_keys, mismatched_keys, error_msgs = bidiff._load_pretrained_model(
            bidiff,
            state_dict,
            pt_path,
            args.controlnet_model_path,
            ignore_mismatched_sizes=False,
        )
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "bidiff"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = BidiffModel.from_pretrained(input_dir, subfolder="bidiff")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            bidiff.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        bidiff.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )
    if accelerator.unwrap_model(bidiff).dtype != torch.float32:
        raise ValueError(
            f"Bidiff loaded as datatype {accelerator.unwrap_model(bidiff).dtype}. {low_precision_error_string}"
        )
    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    param_control = []
    param_mv = []
    param_3d = []
    for name, params in bidiff.named_parameters():
        if name.split('.')[0] == 'unet':
            continue
        elif name.split('.')[0] in ['denoiser3d']:
            if name.split('.')[1] == 'xm':
                continue
            else:
                param_mv.append(params)
        elif name.split('.')[0] == 'model_3d':
            if "gate" in name or "adapter" in name:
                param_3d.append(params)
            else:
                continue
        else:
            param_control.append(params)
            # print("---------control param---------", name, params.requires_grad)
            # print(name, params.requires_grad)
        
    # params_to_optimize = bidiff.parameters()
    params_to_optimize = []
    if len(param_mv) > 0:
        params_to_optimize.append({'params': param_mv, 'lr': args.mv_learning_rate})
    if len(param_3d) > 0:
        params_to_optimize.append({'params': param_3d, 'lr': args.shape_learning_rate})
    if len(param_control) > 0:
        params_to_optimize.append({'params': param_control})

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if text_encoder is not None:
        text_encoder.to(accelerator.device) # , dtype=weight_dtype

    # compute neg prompts
    if args.pre_compute_text_embeddings:
        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")
    else:
        validation_prompt_negative_prompt_embeds = None

    # Dataset
    dataset_cfg = cfg.data.train_set
    if cfg.model.get('use_refer_img', False):
        assert dataset_cfg.get('refer_view_id', None) == cfg.model.refer_view_id
    pre_tokenize = dataset_cfg.get('pre_tokenize', False)
    pre_prompt_encode = dataset_cfg.get('pre_prompt_encode', False)
    prompt_process_args = {}
    if pre_tokenize:
        prompt_process_args['tokenizer'] = tokenizer
        if pre_prompt_encode:
            prompt_process_args['text_encoder'] = text_encoder
    train_dataset = BidiffDataset(**dataset_cfg, **prompt_process_args)

    if args.pre_compute_text_embeddings:
        tokenizer, text_encoder = None, None
        del prompt_process_args
        gc.collect()
        torch.cuda.empty_cache()
    else:
        NotImplementedError
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    bidiff, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        bidiff, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    anneal_start_lod0=0 # NOTE(lihe): hard code for now
    anneal_end_lod0=1000
    for epoch in range(first_epoch, args.num_train_epochs):
        bidiff.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(bidiff):
                # caculate alpha_inter_ratio_lod0
                alpha_inter_ratio_lod0 = get_alpha_inter_ratio(start=anneal_start_lod0, end=anneal_end_lod0, step=step)
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                mv_images = batch['mv_images'].to(dtype=weight_dtype) # (b,v,c,*sp)
                mv_depths = batch['mv_depths'].to(dtype=weight_dtype) # (b,v,c,*sp)
                b, v, c0, h0, w0 = mv_images.shape
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
                voxels = batch.get('voxels', None)
                gt_mesh = batch.get('gt_mesh', None)
                # 3d latents
                latents_3d = batch.get('latents', None)
                gt_sdf = batch.get('gt_sdf', None)
                pos_embed = batch.get('pos_embed', None)
                mv_images_high = batch.get('mv_images_high', None)
                color_voxels = batch.get('color_voxels', None)
                color_noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="exp", coefficient = -20.0)
                sample_rays=dict(rays_d=rays_d, rays_o=rays_o, near_fars=near_fars, K=K, c2ws=c2ws, w2cs=w2cs, affine_mat=affine_mat,
                                 mv_images=mv_images, mv_depths=mv_depths,
                                 extra_rays_d=extra_rays_d, extra_rays_o=extra_rays_o, extra_near_fars=extra_near_fars, extra_K=extra_K, extra_c2ws=extra_c2ws, extra_w2cs=extra_w2cs, extra_affine_mat=extra_affine_mat,
                                 extra_mv_images=extra_mv_images, extra_mv_depths=extra_mv_depths,
                                 alpha_inter_ratio_lod0=alpha_inter_ratio_lod0, step=global_step,
                                 noise_scheduler=noise_scheduler,
                                 latents_3d=latents_3d,
                                 gt_sdf=gt_sdf,
                                 voxels=voxels,
                                 pos_embed=pos_embed,
                                 gt_mesh=gt_mesh,
                                 mv_images_high=mv_images_high,
                                 color_voxels=color_voxels,
                                 color_noise_scheduler=color_noise_scheduler,
                                 )

                model_input = mv_images
                if h0 > 64:
                    model_input = F.interpolate(model_input.view(b * v, c0, h0, w0), size=(64, 64), mode='bilinear', align_corners=False)
                    c, h, w = c0, 64, 64 # hard code for IF
                else:
                    c, h, w = c0, h0, w0
                
                # NOTE(lihe): feed clean images to mv converter to debug!
                sample_rays['clean_images'] = model_input.reshape(b, v, c, h, w)

                # Sample noise that we'll add to the model_input
                model_input = model_input.view(b, v * c, h, w)
                noise = torch.randn_like(model_input)
                # timesteps = torch.tensor([999 - global_step % 1000] * b, device=noise.device, dtype=torch.long)
                if args.max_timestep < noise_scheduler.config.num_train_timesteps:
                    print("="*20)
                    print(f" SET MAX TIMESTEPS TO {args.max_timestep} instead of origin {noise_scheduler.config.num_train_timesteps}")
                    print("="*20)
                    timesteps = torch.randint(0, args.max_timestep, (b,), device=model_input.device)
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=model_input.device)
                timesteps = timesteps.long()

                noisy_latents = sample_rays['noise_scheduler'].add_noise(model_input, noise, timesteps)
                noisy_latents = noisy_latents.view(b, v, c, h, w)
                
                # Get the text embedding for conditioning
                if pre_prompt_encode:
                    prompt_embeds = batch["embed"].to(dtype=weight_dtype) # , device=model_input.device
                    if prompt_embeds.shape[1] == 1:
                        prompt_embeds = prompt_embeds.repeat(1, v, 1, 1)
                else:
                    raise NotImplementedError
                
                validation_prompt_negative_prompt_embeds = validation_prompt_negative_prompt_embeds.to(dtype=weight_dtype)
                if args.classifier_free_guidance_training_rate > 0. and random.random() < args.classifier_free_guidance_training_rate:
                    prompt_embeds_control = validation_prompt_negative_prompt_embeds.unsqueeze(1).expand_as(prompt_embeds)
                    prompt_embeds_unet = prompt_embeds_control
                else:
                    prompt_embeds_control = prompt_embeds
                    prompt_embeds_unet = prompt_embeds_control
                    
                # Forward

                # output = bidiff(
                #     noisy_latents,
                #     timesteps,
                #     prompt_embeds=prompt_embeds,
                # ).sample
                if model_cfg.get('use_refer_img', False):
                    refer_view_id = model_cfg.get('refer_view_id', 5)
                    sample_rays['refer_img'] = model_input.view(b, v, c, h, w)[:, refer_view_id]

                output = bidiff(
                    noisy_latents,
                    timesteps.unsqueeze(1).repeat(1, v).view(-1), # expand to b*nv
                    encoder_hidden_states=prompt_embeds_control, # (b,v, *shape)
                    unet_encoder_hidden_states=prompt_embeds_unet, # (b,v, *shape)
                    sample_rays=sample_rays,
                    weight_dtype=weight_dtype,
                    unet=unet,
                    prompt_3d=batch['caption'],
                )

                mv_losses = output.losses
                model_pred = output.model_pred
                if model_pred.shape[1] == 6: # hard code for IF
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)
                    output.model_pred = model_pred
                else:
                    output.model_pred = model_pred

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                        
                if model_cfg.get('use_refer_img', False):
                    target.view(b, v, c, h, w)[:, refer_view_id] = 0.
                    timesteps_ = timesteps.unsqueeze(1).repeat(1, v)
                    timesteps_[:, refer_view_id] = 0
                    timesteps_ = timesteps_.view(-1)
                else:
                    timesteps_ = timesteps.unsqueeze(1).repeat(1, v).view(-1)
                
                loss = get_loss(output, target, snr_gamma=args.snr_gamma, 
                                timesteps=timesteps_,
                                noise_scheduler=noise_scheduler)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = bidiff.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                    if global_step % args.validation_steps == 0:
                        # compute neg prompts
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            bidiff,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            prompt=batch['caption'],
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=validation_prompt_negative_prompt_embeds,
                            gt_images=batch['mv_images'],
                            sample_rays=sample_rays,
                            num_views=v,
                            unet=unet,
                            noise_scheduler=noise_scheduler,
                        )
                        bidiff.train()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs.update(mv_losses)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bidiff = accelerator.unwrap_model(bidiff)
        bidiff.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

def get_loss(output, target, snr_gamma=None, timesteps=None, noise_scheduler=None):
    model_pred = output.model_pred
    target = target.view(*model_pred.shape)

    if snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(timesteps, noise_scheduler)
        mse_loss_weights = (
            torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        # We first calculate the original loss. Then we mean over the non-batch dimensions and
        # rebalance the sample-wise losses with their respective loss weights.
        # Finally, we take the mean of the rebalanced loss.
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
    loss_trans = output.loss_trans
    loss = loss + loss_trans
    return loss

def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr
    
@torch.no_grad()
def log_validation(
        vae, 
        text_encoder, 
        tokenizer, 
        bidiff, 
        args, 
        accelerator, 
        weight_dtype, 
        step, 
        prompt=None, 
        prompt_embeds=None, 
        negative_prompt_embeds=None,
        gt_images=None,
        sample_rays=None,
        num_views=8,
        unet=None,
        noise_scheduler=None,
    ):
    logger.info("Running validation... ")

    bidiff.eval()
    bidiff = accelerator.unwrap_model(bidiff)
    pipeline_args = {}
    if vae is not None:
        pipeline_args['vae'] = vae
    
    pipeline = BidiffPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        bidiff=bidiff,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
        local_files_only=True,
        **pipeline_args
    )
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type
    
    if bidiff.dpm_solver_3d:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    b, v, c, h, w = sample_rays['clean_images'].shape
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.pre_compute_text_embeddings:
        b, v, *_ = prompt_embeds.shape
        negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1).repeat(b, v, 1, 1)
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # images = []
    with torch.autocast("cuda", enabled=weight_dtype!=torch.float32):
        images = pipeline(**pipeline_args, height=args.resolution, width=args.resolution, 
                        generator=generator, sample_rays=sample_rays, unet=unet,
                        weight_dtype=weight_dtype, num_views=num_views, output_type='pt',
                        tokenizer_max_length=args.tokenizer_max_length, 
                        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                        noise_scheduler=noise_scheduler, num_inference_steps=args.num_inference_steps,
                        prompt_3d=prompt, # NOTE(lihe): also feed prompt to pipeline3d
                        # clean_image=sample_rays['clean_images'].view(b * v, c, h, w)
                        ).images # (b*v,c,h,w)
    
    bv, *chw = images.shape
    dir_path = os.path.join(args.output_dir, 'sample_images')
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, 'step_{:06d}_sample.png'.format(step))
    torchvision.utils.save_image(images, path, nrow=4)
    if gt_images is not None:
        res_gt = gt_images.shape[-1]
        gt_images = gt_images.view(bv, c, res_gt, res_gt)
        gt_images = (gt_images / 2 + 0.5).clamp(0, 1)
        gt_path = os.path.join(dir_path, 'step_{:06d}_ground_truth.png'.format(step))
        torchvision.utils.save_image(gt_images, gt_path, nrow=4)
    prompt_path = os.path.join(dir_path, 'step_{:06d}_prompt.txt'.format(step))
    with open(prompt_path, 'w') as f:
        if isinstance(prompt, list):
            for p in prompt:
                if isinstance(p, list):
                    for p_ in p:
                        f.write(p_ + '\n')
                else:
                    f.write(p + '\n')

    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in images])
    #         tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
    #     if tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 "validation": [
    #                     wandb.Image(image, caption=f"{i}: {prompt}") for i, image in enumerate(images)
    #                 ]
    #             }
    #         )

    del pipeline
    torch.cuda.empty_cache()
    # bidiff.train()

    return images


if __name__ == '__main__':
    args = parse_args()
    main(args)
