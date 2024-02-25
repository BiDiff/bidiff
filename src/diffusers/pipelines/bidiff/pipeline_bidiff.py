import html
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import re
import urllib.parse as ul
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers, DDPMScheduler, DPMSolverMultistepScheduler
from ...models.bidiff import BidiffModel

from diffusers.models.shap_e.shap_e.diffusion.sample import sample_latents
from diffusers.models.shap_e.shap_e.models.download import load_model, load_config
from diffusers.models.shap_e.shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from torchvision.utils import save_image

from ...utils import (
    BACKENDS_MAPPING,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    is_bs4_available,
    is_ftfy_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from ..stable_diffusion import StableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""


class BidiffPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):

    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    bidiff: BidiffModel
    scheduler: DDPMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[StableDiffusionSafetyChecker]
    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor"]


    def __init__(
        self,
        text_encoder: T5Tokenizer,
        tokenizer: T5EncoderModel,
        bidiff: BidiffModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        vae: AutoencoderKL = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )
        
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        if isinstance(bidiff, (list, tuple)):
            raise NotImplementedError
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            bidiff=bidiff,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        
        if self.vae is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 1
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        if self.vae is not None:
            self.vae.enable_slicing()
        else:
            raise NotImplementedError

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if self.vae is not None:
            self.vae.disable_slicing()
        else:
            raise NotImplementedError

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        if self.vae is not None:
            self.vae.enable_tiling()
        else:
            raise NotImplementedError

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if self.vae is not None:
            self.vae.disable_tiling()
        else:
            raise NotImplementedError

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.bidiff, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)
        
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None

        if self.text_encoder is not None:
            _, hook = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=hook)

            # Accelerate will move the next model to the device _before_ calling the offload hook of the
            # previous model. This will cause both models to be present on the device at the same time.
            # IF uses T5 for its text encoder which is really large. We can manually call the offload
            # hook for the text encoder to ensure it's moved to the cpu before the unet is moved to
            # the GPU.
            self.text_encoder_offload_hook = hook
        
        if self.vae is not None:
            cpu_offload_with_hook(self.vae, device, prev_module_hook=hook)

        _, hook = cpu_offload_with_hook(self.bidiff, device, prev_module_hook=hook)

        # if the safety checker isn't called, `unet_offload_hook` will have to be called to manually offload the unet
        self.unet_offload_hook = hook

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def remove_all_hooks(self):
        if is_accelerate_available():
            from accelerate.hooks import remove_hook_from_module
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        for model in [self.text_encoder, self.bidiff, self.safety_checker]:
            if model is not None:
                remove_hook_from_module(model, recurse=True)

        self.unet_offload_hook = None
        self.text_encoder_offload_hook = None
        self.final_offload_hook = None

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.bidiff, "_hf_hook"):
            return self.device
        for module in self.bidiff.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device
    
    @torch.no_grad()
    def _encode_prompt(
        self,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance=True,
        prompt=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clean_caption: bool = False,
        tokenizer=None,
        tokenizer_max_length=None,
        text_encoder=None,
        text_encoder_use_attention_mask=False,
        num_views=8,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            
        if device is None:
            device = self._execution_device
        
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            raise NotImplementedError
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) // num_views
        else:
            batch_size, num_views = prompt_embeds.shape[:2]

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            if tokenizer is None:
                tokenizer = self.tokenizer
            if text_encoder is None:
                text_encoder = self.text_encoder
            
            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, tokenizer)
                # raise NotImplementedError

            if tokenizer_max_length is not None:
                max_length = tokenizer_max_length
            else:
                max_length = tokenizer.model_max_length
                raise NotImplementedError

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
                raise NotImplementedError

            if (hasattr(text_encoder.config, "use_attention_mask") and \
                text_encoder.config.use_attention_mask) or text_encoder_use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
                raise NotImplementedError

            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            if prompt_embeds.shape[0] == num_views:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            else:
                NotImplementedError

        if text_encoder is not None:
            dtype = text_encoder.dtype
        elif self.bidiff is not None:
            dtype = self.bidiff.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, num_views, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, num_views, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[2]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            if (hasattr(text_encoder.config, "use_attention_mask") and \
                text_encoder.config.use_attention_mask) or text_encoder_use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
                raise NotImplementedError

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1).repeat(batch_size, num_views, 1, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[2]

            negative_prompt_embeds = negative_prompt_embeds.to(device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, num_views, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        if len(latents.shape) == 5:
            b, v, c, h, w = latents.shape
            latents = latents.view(b * v, c, h, w)
        if self.vae is not None:
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents
            raise NotImplementedError
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    
    def prepare_intermediate_images(self, batch_size, num_views, num_channels, height, width, dtype, device, generator):
        shape = (batch_size, num_views, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        intermediate_images = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_views, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_views, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # shape = (batch_size, num_views, num_channels_latents, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 25, # 25,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        unet=None,
        tokenizer=None,
        tokenizer_max_length=None,
        text_encoder=None,
        text_encoder_use_attention_mask=False,
        sample_rays=None,
        weight_dtype=None,
        num_views=None,
        noise_scheduler=None,
        clean_image=None,
        prompt_3d=None,
        cond_guidance=True,
        cond_guidance_interval=80,
        control_guidance_interval=1000,
        cond_decouple=False,
        cond_guidance_scale=7.0,
        text_guidance_scale=7.0,
        mesh_save_path=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) // num_views
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        global_pool_conditions = self.bidiff.config.global_pool_conditions
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
            text_encoder=text_encoder,
            text_encoder_use_attention_mask=text_encoder_use_attention_mask,
            num_views=num_views,
        )
        if cond_decouple:
            nega_embed, embed = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([nega_embed, embed, embed])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        inputs = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            num_views,
            self.bidiff.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        if self.bidiff.input_res > 64:
            inputs_hr = self.prepare_intermediate_images(
                batch_size * num_images_per_prompt,
                num_views,
                self.bidiff.config.in_channels,
                self.bidiff.input_res,
                self.bidiff.input_res,
                prompt_embeds.dtype,
                device,
                generator,
            )
        else:
            inputs_hr = None
        
        # NOTE(lihe): add sdf gen
        if self.bidiff.sdf_gen:
            noisy_sdf_input = torch.randn([batch_size, 96*96*96, 1], device=inputs.device)
        else:
            noisy_sdf_input = None
        pred_clean_sdf = None
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()
        
        # NOTE(lihe): if use dpm solver sample 3d prior
        if self.bidiff.dpm_solver_3d and isinstance(self.scheduler, DPMSolverMultistepScheduler):
            scheduler_args = {}
            scheduler_args["variance_type"] = "fixed_small"
            noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='exp', prediction_type="sample")
            dpm_solver_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config, **scheduler_args)
            print("=== dpm_solver_3d inherit timesteps from 2D ===", timesteps)
            dpm_solver_scheduler.set_timesteps(num_inference_steps, device=device, sync_timesteps=timesteps)
            timesteps_3d = dpm_solver_scheduler.timesteps
        else:
            dpm_solver_scheduler = None
        
        if self.bidiff.sdf_gen and isinstance(self.scheduler, DPMSolverMultistepScheduler):
            scheduler_args = {}
            scheduler_args["variance_type"] = "fixed_small"
            noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type="sample") # NOTE: squaredcos_cap_v2
            sdf_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config, **scheduler_args)
            sdf_scheduler.set_timesteps(num_inference_steps, device=device, sync_timesteps=timesteps)
            timesteps_sdf = sdf_scheduler.timesteps
        elif self.bidiff.sdf_gen:
            sdf_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type="sample")
            sdf_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps_sdf = sdf_scheduler.timesteps
        else:
            sdf_scheduler = None

        # 7. Denoising loop
        # expand sample rays
        if do_classifier_free_guidance:
            guid_repeat = 3 if cond_decouple else 2
            for key in sample_rays.keys():
                if sample_rays[key] is not None and key not in ['step', 'alpha_inter_ratio_lod0', 'noise_scheduler', 'voxels', 'feature_volume_save_path', 'feature_volume_load_path', 'gt_mesh']:
                # if isinstance(sample_rays[key], torch.Tensor):
                    sample_rays[key] = sample_rays[key].repeat([guid_repeat]+[1]*(len(sample_rays[key].shape) - 1))
                if key == 'voxels' and sample_rays[key] is not None: # voxels is list
                    sample_rays[key] = sample_rays[key] * guid_repeat
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if dpm_solver_scheduler is not None:
                    assert t == timesteps_3d[i], f"2D sampling steps {timesteps} must match the 3d steps {timesteps_3d}."
                if sdf_scheduler is not None:
                    assert t == timesteps_sdf[i], f"2D sampling steps {timesteps} must match the sdf steps {timesteps_sdf}."
                if hasattr(self.bidiff, 'denoiser3d') and hasattr(self.bidiff.denoiser3d, 'sdf_def_network'):
                    self.bidiff.denoiser3d.sdf_def_network.train()
                    self.bidiff.denoiser3d.sdf_def_network.training = False
                
                #NOTE(lihe): when t > 700, just use clean x0 + noise as inputs and use clean x0 as neus_pred_x0
                replace_cond_img_by_neus = False
                if self.bidiff.config.lazy_t is not None:
                    if t >= self.bidiff.lazy_t and i != 0:
                        clean_x0 = sample_rays['mv_images'][1].view(batch_size * num_images_per_prompt * num_views, 3, 64, 64)
                        noise = torch.randn_like(clean_x0)
                        t_batch = t.repeat(batch_size * num_images_per_prompt * num_views).view(-1)
                        inputs = sample_rays['noise_scheduler'].add_noise(clean_x0, noise, t_batch)
                        inputs = inputs.view(batch_size * num_images_per_prompt, num_views, 3, 64, 64)
                        neus_pred_x0 = sample_rays['mv_images']
                        print("--- not start yet ---")
                    elif t < self.bidiff.lazy_t and replace_cond_img_by_neus:
                        print("=== replace_cond_img_by_neus ===")
                        sample_rays['mv_images'] = neus_pred_x0.view(-1, num_views, 3, 64, 64)

                # double inputs
                if do_classifier_free_guidance:
                    guid_repeat = 3 if cond_decouple else 2
                    model_input = torch.cat([inputs] * guid_repeat)
                    t_batch = t.repeat(batch_size * num_images_per_prompt * num_views * guid_repeat).view(-1)
                    if inputs_hr is not None:
                        model_input_hr = torch.cat([inputs_hr] * guid_repeat)
                        model_input_hr = self.scheduler.scale_model_input(model_input_hr, t)
                else:
                    model_input = inputs
                    t_batch = t.repeat(batch_size * num_images_per_prompt * num_views).view(-1)
                    if inputs_hr is not None:
                        model_input_hr = inputs_hr
                        model_input_hr = self.scheduler.scale_model_input(model_input_hr, t)

                model_input = self.scheduler.scale_model_input(model_input, t)
                
                # NOTE(lihe): occ diff
                if do_classifier_free_guidance and self.bidiff.sdf_gen:
                    noisy_sdf = torch.cat([noisy_sdf_input]*guid_repeat)
                    if pred_clean_sdf is not None:
                        pred_clean_sdf = torch.cat([pred_clean_sdf]*guid_repeat)
                else:
                    assert do_classifier_free_guidance, 'currently we should use classifier free guidance'
                    noisy_sdf = None

                if self.bidiff.use_3d_prior and self.bidiff.skip_denoise and i==0:
                    # skip connect neus denoise
                    neus_noisy_latents_inputs = inputs
                    neus_noisy_latents = model_input
                elif self.bidiff.use_3d_prior and self.bidiff.skip_denoise and i != 0:
                    model_neus_pred_x0 = neus_pred_x0.view(guid_repeat, num_views, 3, 64, 64)[1]
                    noise = torch.randn_like(model_neus_pred_x0)
                    t_batch_skip = t.repeat(batch_size * num_images_per_prompt * num_views).view(-1)
                    neus_noisy_latents_inputs = sample_rays['noise_scheduler'].add_noise(model_neus_pred_x0, noise, t_batch_skip)
                    neus_noisy_latents = torch.cat([neus_noisy_latents_inputs]*guid_repeat)
                    neus_noisy_latents = self.scheduler.scale_model_input(neus_noisy_latents, t)
                else:
                    neus_noisy_latents = None

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = inputs
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    unet_prompt_embeds = prompt_embeds.chunk(guid_repeat)[-1]
                else:
                    control_model_input = model_input
                    unet_prompt_embeds = prompt_embeds.clone()

                
                if do_classifier_free_guidance:
                    if cond_guidance_interval >= 1000:
                        cond_guidance = False
                    elif t % cond_guidance_interval < (500 // num_inference_steps) or \
                        t % cond_guidance_interval > (cond_guidance_interval - 500 // num_inference_steps):
                        guidance_scale = cond_guidance_scale
                    else:
                        guidance_scale = text_guidance_scale
                
                # NOTE(lihe): add 3d prior
                if self.bidiff.use_3d_prior and i==0:
                    noisy_latents_3d = torch.randn([batch_size, 1024*1024], device=model_input.device)
                    noisy_latents_3d = torch.cat([noisy_latents_3d]*guid_repeat) # NOTE(lihe): fixed bug !
                elif not self.bidiff.use_3d_prior:
                    noisy_latents_3d = None
                
                # NOTE(lihe): implement lazy noise strategy
                lazy_from_prior = True
                if self.bidiff.lazy_3d and lazy_from_prior and self.bidiff.use_3d_prior and i==0:
                    model = load_model('text300M', device=device)
                    assert len(prompt_3d) == 1, "we now only support sample one object"
                    prompt_to_text300M = prompt_3d[0][0].split('.')[2] # [p[0].split('.')[2] for p in prompt_3d]
                    with torch.no_grad():
                        latents = sample_latents(
                                        batch_size=batch_size,
                                        model=model,
                                        diffusion=self.bidiff.denoiser3d.ddpm_3d,
                                        guidance_scale=guidance_scale,
                                        model_kwargs=dict(texts=[prompt_to_text300M] * batch_size),
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
                        render_mode = 'nerf' # you can change this to 'stf'
                        size = 64 # this is the size of the renders; higher values take longer to render.
                        cameras = create_pan_cameras(size, device)
                        images, img_tensor = decode_latent_images(self.bidiff.denoiser3d.xm, latents[0], cameras, rendering_mode=render_mode, return_tensor=True)
                        print(f"============saving lazy clean gif of prompt {prompt_to_text300M}==============")
                        images[0].save('debug/debug_lazy_clean.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
                        # print("======== img_tensor shape =======", img_tensor.shape) # [1, 20, 64, 64, 3]
                        img_tensor = img_tensor.permute(0,1,4,2,3)
                        # img_tensor = img_tensor[:, :8] * 2 - 1.
                        img_tensor = img_tensor[:, 2:10] * 2 - 1.
                        # img_tensor = torch.cat([img_tensor[None]]*2) # [2, 8, 3, 64, 64]
                        img_tensor = img_tensor.repeat(guid_repeat, 1, 1, 1, 1)
                        sample_rays['mv_images'] = img_tensor 
                        noisy_latents_3d = torch.cat([latents]*guid_repeat)
                        lazy_t = self.bidiff.denoiser3d.lazy_t
                        noisy_latents_3d = self.bidiff.denoiser3d.ddpm_3d.q_sample(noisy_latents_3d, torch.tensor(lazy_t).repeat(noisy_latents_3d.shape).to(noisy_latents_3d.device))
                        save_image((img_tensor[0] + 1)/2., 'debug/debug_lazy_render_from_text.png', nrow=4)
                        del model
                        del cameras
                        del images
                
                output = self.bidiff(
                    model_input,
                    t_batch, # expand to b*nv
                    encoder_hidden_states=prompt_embeds,
                    unet_encoder_hidden_states=unet_prompt_embeds,
                    sample_rays=sample_rays,
                    weight_dtype=weight_dtype,
                    unet=unet,
                    noisy_latents_3d=noisy_latents_3d,
                    prompt_3d=prompt_3d,
                    new_batch=i==0, # bidiff will update the 3d prior prompt if new_batch==True
                    dpm_solver_scheduler=dpm_solver_scheduler,
                    neus_noisy_latents=neus_noisy_latents,
                    noisy_sdf=noisy_sdf,
                    cond_guidance=cond_guidance,
                    cond_guidance_interval=cond_guidance_interval,
                    control_guidance_interval=control_guidance_interval,
                    num_inference_steps=num_inference_steps,
                    mesh_save_path=mesh_save_path,
                    cond_decouple=cond_decouple,
                    noisy_latents_hr=model_input_hr if inputs_hr is not None else None,
                    pred_clean_sdf=pred_clean_sdf, # NOTE(lihe): also feed pred clean sdf
                )

                noise_pred = output.model_pred

                if self.bidiff.use_3d_prior:
                    noisy_latents_3d = output.noisy_latents_3d_prev
                    neus_pred_x0 = output.neus_pred_x0

                if len(noise_pred.shape) == 5:
                    _b, v, *chw = noise_pred.shape
                    noise_pred = noise_pred.view(_b * v, *chw)
                
                if len(model_input.shape) == 5:
                    _b, v, *chw = model_input.shape
                    _bv = _b * v
                else:
                    _bv, *chw = model_input.shape
                model_input = model_input.view(_bv, *chw)

                # perform guidance
                if do_classifier_free_guidance:
                    if cond_decouple:
                        noise_pred_no_text, noise_pred_no_consist, noise_pred_normal = noise_pred.chunk(guid_repeat)
                        if noise_pred_normal.shape[1] == 6: # hard code for IF
                            noise_pred_no_text, _ = noise_pred_no_text.split(model_input.shape[1], dim=1)
                            noise_pred_no_consist, _ = noise_pred_no_consist.split(model_input.shape[1], dim=1)
                            noise_pred_normal, predicted_variance = noise_pred_normal.split(model_input.shape[1], dim=1)
                            noise_pred = noise_pred_normal + \
                                (text_guidance_scale - 1.0) * (noise_pred_normal - noise_pred_no_text) + \
                                (cond_guidance_scale - 1.0) * (noise_pred_normal - noise_pred_no_consist)
                            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                        else:
                            noise_pred = noise_pred_normal + \
                                (text_guidance_scale - 1.0) * (noise_pred_normal - noise_pred_no_text) + \
                                (cond_guidance_scale - 1.0) * (noise_pred_normal - noise_pred_no_consist)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        if noise_pred_text.shape[1] == 6:
                            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                        else:
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # perform guidance for sdf gen
                if do_classifier_free_guidance and self.bidiff.sdf_gen:
                    pred_clean_sdf = output.pred_clean_sdf
                    if cond_decouple:
                        pred_clean_sdf_uncond, _, pred_clean_sdf_text = pred_clean_sdf.chunk(guid_repeat)
                    else:
                        pred_clean_sdf_uncond, pred_clean_sdf_text = pred_clean_sdf.chunk(2)
                    # pred_clean_sdf = pred_clean_sdf_uncond + guidance_scale * (pred_clean_sdf_text - pred_clean_sdf_uncond)
                    # NOTE(lihe): dont use guidance for debugging
                    pred_clean_sdf = pred_clean_sdf_text
                        
                if "variance_type" in self.scheduler.config:
                    if self.scheduler.config.variance_type not in ["learned", "learned_range"] and noise_pred.shape[1] == 6:
                        noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
                
                # compute the previous noisy sample x_t -> x_t-1
                inputs = inputs.view(batch_size * num_images_per_prompt * num_views, *chw)
                model_input = self.scheduler.step(
                    noise_pred, t, 
                    inputs if not self.bidiff.skip_denoise else neus_noisy_latents_inputs.view(batch_size * num_images_per_prompt * num_views, *chw), 
                    **extra_step_kwargs, return_dict=False
                )[0]

                if inputs_hr is not None:
                    pred_x0 = self.scheduler.step_w_device(
                        noise_pred, t.repeat(batch_size * num_images_per_prompt * num_views), 
                        inputs if not self.bidiff.skip_denoise else neus_noisy_latents_inputs.view(batch_size * num_images_per_prompt * num_views, *chw),
                        device=noise_pred.device
                    )
                    pred_x0_hr = F.interpolate(pred_x0, size=(self.bidiff.input_res, self.bidiff.input_res), mode='bilinear', align_corners=False)
                    inputs_hr = self.scheduler.add_noise_x0_custom_t(
                        pred_x0_hr, t, 
                        inputs_hr if not self.bidiff.skip_denoise else neus_noisy_latents_inputs.view(batch_size * num_images_per_prompt * num_views, *chw),
                    ).view(*inputs_hr.shape)
                
                if self.bidiff.sdf_gen and i < len(timesteps) - 1:
                    time = t.repeat(batch_size).view(-1)
                    time_next = timesteps[i+1].repeat(batch_size).view(-1)
                    log_snr = self.bidiff.denoiser3d.log_snr(time/1000.).view(-1, 1, 1)
                    log_snr_next = self.bidiff.denoiser3d.log_snr(time_next/1000.).view(-1, 1, 1)
                    alpha, sigma = self.bidiff.denoiser3d.log_snr_to_alpha_sigma(log_snr)
                    alpha_next, sigma_next = self.bidiff.denoiser3d.log_snr_to_alpha_sigma(log_snr_next)
                        
                    c = -torch.special.expm1(log_snr - log_snr_next)
                    mean = alpha_next * (pred_clean_sdf * (1 - c) / alpha + c * pred_clean_sdf)
                    variance = (sigma_next ** 2) * c
                    noise = torch.randn_like(pred_clean_sdf)
                    noisy_sdf_input = mean + torch.sqrt(variance) * noise

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, model_input)

                inputs = model_input.view(batch_size * num_images_per_prompt, num_views, *chw)

                
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.bidiff.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent" and hasattr(self, 'vae') and self.vae is not None:
            image = self.vae.decode(model_input / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])
            has_nsfw_concept = None
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        elif output_type == "latent":
            image = model_input
            has_nsfw_concept = None
        elif output_type == "pil":
            image = model_input
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            has_nsfw_concept = None
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)

            # 11. Apply watermark
            # if self.watermarker is not None:
            #     image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == 'pt':
            image = model_input
            image = (image / 2 + 0.5).clamp(0, 1)
            has_nsfw_concept = None
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            raise NotImplementedError
        
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)