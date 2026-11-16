import argparse
import io
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import load_image
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision.transforms import v2
from tqdm.auto import tqdm

from utils import (
    coc_path_from_rgb,
    disparity_path_from_depth,
    disparity_to_image,
    encode_prompt,
    encode_stage1_coc,
    encode_vae,
    fit_long_edge_and_pad,
    load_float_map,
    pack_mask,
    prune_checkpoints,
)

logger = get_logger(__name__)

# Fixed offset added to the flow-matching noise.
OFFSET_NOISE = 0.1
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage 1 LoRA (CoC and disparity).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/train")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--use_lora_bias", action="store_true")
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--gaussian_init_lora", action="store_true")
    parser.add_argument("--train_norm_layers", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_validation_images", type=int, default=5)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--tracker_project_name", type=str, default="AnyBokeh-stage1")
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--upcast_before_saving", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--data_file", type=str, required=True, help="JSONL list relative to --dataset_root, for example data_lists/stage1_train.jsonl.")
    parser.add_argument("--val_data_file", type=str, default=None, help="JSONL list relative to --dataset_root, for example data_lists/stage1_val.jsonl.")
    parser.add_argument("--dataset_root", type=str, required=True, help="UnrealBokeh directory. --data_file and the paths inside that list are relative to it.")
    parser.add_argument(
        "--global_text_prompt",
        type=str,
        default="Predict the Circle-of-Confusion map and disparity map of the given image.",
    )
    parser.add_argument("--weighted_loss_alpha", type=float, default=10.0, help="Extra weight on large CoC values.")
    parser.add_argument("--bg_boost_ratio", type=float, default=1.5, help="Extra weight on background (positive CoC).")
    parser.add_argument("--weighted_loss_for_disp_alpha", type=float, default=10.0, help="Extra weight on extreme disparity.")
    args = parser.parse_args()
    return args


class Stage1Dataset(torch.utils.data.Dataset):
    """Long-edge resize, pad to a square, and pack [image | CoC | disparity].

    `data_file` and each entry's `rgb_file` and `depth_file` are relative to the UnrealBokeh root.
    CoC is `coc/<same stem>.npy`. Disparity is `disparity/<depth stem>.npy`.
    """

    def __init__(self, data_file, resolution, dataset_root, deterministic=False):
        if not os.path.isabs(data_file):
            data_file = os.path.join(dataset_root, data_file)
        with open(data_file, "r", encoding="utf-8") as handle:
            self.data = [json.loads(line) for line in handle if line.strip()]
        self.resolution = resolution
        self.deterministic = deterministic
        self.root = dataset_root

    def __len__(self):
        return len(self.data)

    def _path(self, path):
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        item = self.data[index]
        rgb_path = self._path(item["rgb_file"])
        depth_path = self._path(item["depth_file"])
        image = load_image(rgb_path)
        coc = Image.fromarray(load_float_map(coc_path_from_rgb(rgb_path)))
        disp = Image.fromarray(load_float_map(disparity_path_from_depth(depth_path)))
        if not self.deterministic and random.random() > 0.5:
            image, coc, disp = TF.hflip(image), TF.hflip(coc), TF.hflip(disp)
        (image, coc, disp), scale = fit_long_edge_and_pad([image, coc, disp], self.resolution, deterministic=self.deterministic)
        coc = encode_stage1_coc(np.array(coc, dtype=np.float32) * scale)
        disp = disparity_to_image(np.array(disp, dtype=np.float32))
        if not self.deterministic:
            image = self._augment_rgb(image)

        resolution = self.resolution
        canvas = Image.new("RGB", (3 * resolution, resolution))
        canvas.paste(image, (0, 0))
        canvas.paste(coc, (resolution, 0))
        canvas.paste(disp, (2 * resolution, 0))
        mask = torch.zeros(1, resolution, 3 * resolution)
        mask[:, :, resolution:] = 1
        values = TF.to_tensor(canvas)
        masked = values * (1 - mask)
        return {"image": values * 2 - 1, "masked_image": (masked * 2 - 1).unsqueeze(0), "masks": mask}

    def _augment_rgb(self, image):
        if random.random() > 0.5:
            image = v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image)
        if random.random() > 0.5:
            image = TF.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = self._poisson(image, random.uniform(50.0, 200.0))
        if random.random() > 0.5:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(40, 95))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        if random.random() > 0.5:
            array = np.array(image, dtype=np.float64)
            image = Image.fromarray(np.clip(array + np.random.normal(0, random.uniform(1.0, 10.0), array.shape), 0, 255).astype(np.uint8))
        if random.random() > 0.3:
            image = self._motion_blur(image)
        return image

    def _poisson(self, image, photon_scale):
        array = np.array(image, dtype=np.float64) / 255.0
        noisy = np.random.poisson(array * photon_scale).astype(np.float64) / photon_scale
        return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))

    def _motion_blur(self, image):
        size = random.choice((3, 5, 7))
        kernel = np.zeros((size, size), dtype=np.float32)
        angle = math.radians(random.uniform(0, 180))
        center = size // 2
        for t in np.linspace(-0.5, 0.5, size * 2):
            x = int(round(center + t * (size - 1) * math.cos(angle)))
            y = int(round(center + t * (size - 1) * math.sin(angle)))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        if kernel.sum() == 0:
            kernel[center, center] = 1.0
        kernel /= kernel.sum()
        array = np.array(image.convert("RGB"), dtype=np.float32)
        weight = torch.from_numpy(kernel).view(1, 1, size, size)
        channels = [F.conv2d(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)[:, c:c + 1], weight, padding=size // 2) for c in range(3)]
        out = torch.cat(channels, dim=1).squeeze(0).permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([sample["image"] for sample in batch]),
        "masked_images": torch.stack([sample["masked_image"] for sample in batch]),
        "masks": torch.stack([sample["masks"] for sample in batch]),
    }


def stage1_loss_weight(pixel_values, latent_height, alpha, bg_boost, disp_alpha):
    """Upweight large CoC, background CoC, and extreme disparity. The image panel stays at weight 1."""
    resolution = pixel_values.shape[-2]
    coc = pixel_values[:, :, :, resolution:2 * resolution].mean(dim=1, keepdim=True)
    disp = pixel_values[:, :, :, 2 * resolution:3 * resolution].mean(dim=1, keepdim=True)
    coc_weight = (1.0 + alpha * coc.abs() ** 2) * (1.0 + (bg_boost - 1.0) * (coc > 0).float())
    disp_weight = 1.0 + disp_alpha * 4.0 * (disp - 0.5) ** 2

    def tile(weight):
        weight = F.interpolate(weight, size=(latent_height, latent_height), mode="area")
        return weight / (weight.mean(dim=(1, 2, 3), keepdim=True) + 1e-6) + 1.0

    coc_weight, disp_weight = tile(coc_weight), tile(disp_weight)
    return torch.cat([torch.ones_like(coc_weight), coc_weight, disp_weight], dim=3)


def log_validation(transformer, accelerator, weight_dtype, args, prompt_embeds, pooled, text_ids, dataset, height, width, step):
    pipe = FluxFillPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    count = min(args.num_validation_images, len(dataset))
    images = []
    for index in random.sample(range(len(dataset)), count):
        sample = dataset[index]
        condition = TF.to_pil_image((sample["masked_image"].squeeze(0) * 0.5 + 0.5).clamp(0, 1))
        mask = TF.to_pil_image(sample["masks"].squeeze(0))
        output = pipe(
            prompt=None,
            image=condition,
            mask_image=mask,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            num_inference_steps=20,
            guidance_scale=args.guidance_scale,
            max_sequence_length=512,
        ).images[0]
        images.extend((condition, output))
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb

            tracker.log({"validation": [wandb.Image(image) for image in images]}, step=step)
    del pipe


def main():
    args = parse_args()
    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError("`gaussian` LoRA init is not supported together with `use_lora_bias`.")
    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(Path(args.output_dir, args.logging_dir))),
    )
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision, cache_dir=args.cache_dir
    )
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.to(dtype=torch.float32)
    transformer.to(dtype=weight_dtype, device=accelerator.device)

    if args.train_norm_layers:
        for name, param in transformer.named_parameters():
            if any(key in name for key in NORM_LAYER_PREFIXES):
                param.requires_grad = True
    target_modules = [layer.strip() for layer in args.lora_layers.split(",")] if args.lora_layers else [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    ]
    transformer.add_adapter(LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    ))

    def unwrap(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    def save_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        lora_layers = None
        for model in models:
            if isinstance(unwrap(model), type(unwrap(transformer))):
                lora_layers = get_peft_model_state_dict(unwrap(model))
                if args.train_norm_layers:
                    lora_layers.update({
                        f"transformer.{name}": param
                        for name, param in unwrap(model).named_parameters()
                        if any(key in name for key in NORM_LAYER_PREFIXES)
                    })
            else:
                raise ValueError(f"Unexpected model in save hook: {model.__class__}")
            if weights:
                weights.pop()
        FluxFillPipeline.save_lora_weights(output_dir, transformer_lora_layers=lora_layers)

    def load_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if not isinstance(unwrap(model), type(unwrap(transformer))):
                raise ValueError(f"Unexpected model in load hook: {model.__class__}")
        transformer_ = unwrap(transformer)
        lora_state = FluxFillPipeline.lora_state_dict(input_dir)
        transformer_state = {
            key.replace("transformer.", ""): value
            for key, value in lora_state.items()
            if key.startswith("transformer.") and "lora" in key
        }
        set_peft_model_state_dict(transformer_, transformer_state, adapter_name="default")
        if args.train_norm_layers:
            norm_state = {
                key: value for key, value in lora_state.items()
                if key.startswith("transformer.") and any(prefix in key for prefix in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = FluxFillPipeline._load_norm_into_transformer(
                norm_state, transformer=transformer_, discard_original_layers=False
            )
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [param for param in transformer.parameters() if param.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = Stage1Dataset(args.data_file, args.resolution, args.dataset_root)
    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(math.ceil(len(train_dataloader) / accelerator.num_processes) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(args, weight_dtype, accelerator.device)

    if accelerator.is_main_process and args.report_to != "none":
        init_kwargs = {"wandb": {"name": os.path.basename(os.path.abspath(args.output_dir))}} if args.report_to == "wandb" else None
        accelerator.init_trackers(
            args.tracker_project_name,
            config={key: value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool))},
            init_kwargs=init_kwargs,
        )

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            names = [name for name in os.listdir(args.output_dir) if name.startswith("checkpoint-")]
            names = sorted(names, key=lambda name: int(name.split("-")[1]))
            resume_path = os.path.join(args.output_dir, names[-1]) if names else None
        else:
            resume_path = args.resume_from_checkpoint
        if resume_path is None:
            logger.info("No checkpoint found, starting from scratch.")
        else:
            accelerator.load_state(resume_path)
            global_step = int(os.path.basename(resume_path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed from {resume_path} at step {global_step}")

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    progress = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, reference):
        schedule = noise_scheduler.timesteps.to(reference.device)
        values = noise_scheduler.sigmas.to(device=reference.device, dtype=reference.dtype)
        indices = [(schedule == step).nonzero().item() for step in timesteps]
        sigma = values[indices].flatten()
        while sigma.ndim < reference.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    for _ in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                embeds = prompt_embeds.repeat(batch["pixel_values"].shape[0], 1, 1)
                pooled = pooled_prompt_embeds.repeat(batch["pixel_values"].shape[0], 1)
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    embeds = torch.zeros_like(embeds)
                    pooled = torch.zeros_like(pooled)

                vae.to(accelerator.device)
                pixels = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                masked_pixels = batch["masked_images"].to(accelerator.device, dtype=vae.dtype).reshape(pixels.shape)
                latents = encode_vae(pixels, vae, weight_dtype)
                masked_latents = encode_vae(masked_pixels, vae, weight_dtype)
                if args.offload:
                    vae.cpu()

                noise = torch.randn_like(latents)
                noise = noise + OFFSET_NOISE * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype)
                timestep_weights = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=latents.shape[0],
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (timestep_weights * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(latents.device)
                sigmas = get_sigmas(timesteps, latents)
                noisy = (1.0 - sigmas) * latents + sigmas * noise

                mask = pack_mask(batch["masks"].to(accelerator.device), latents, vae_scale_factor)
                packed_noisy = FluxFillPipeline._pack_latents(noisy, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
                packed_masked = FluxFillPipeline._pack_latents(masked_latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
                packed_mask = FluxFillPipeline._pack_latents(mask, latents.shape[0], vae_scale_factor * vae_scale_factor, latents.shape[2], latents.shape[3])
                hidden = torch.cat((packed_noisy, torch.cat((packed_masked, packed_mask), dim=-1)), dim=2)
                image_ids = FluxFillPipeline._prepare_latent_image_ids(
                    latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2, accelerator.device, weight_dtype
                )
                guidance = None
                if unwrap(transformer).config.guidance_embeds:
                    guidance = torch.full((latents.shape[0],), args.guidance_scale, device=latents.device, dtype=weight_dtype)

                prediction = transformer(
                    hidden_states=hidden.to(weight_dtype),
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled,
                    encoder_hidden_states=embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    return_dict=False,
                )[0]
                prediction = FluxFillPipeline._unpack_latents(
                    prediction,
                    height=latents.shape[2] * vae_scale_factor,
                    width=latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - latents
                per_pixel = weighting.float() * (prediction.float() - target.float()) ** 2
                loss_weight = stage1_loss_weight(
                    batch["pixel_values"],
                    per_pixel.shape[2],
                    args.weighted_loss_alpha,
                    args.bg_boost_ratio,
                    args.weighted_loss_for_disp_alpha,
                )
                loss = (per_pixel * loss_weight.to(per_pixel.device)).reshape(target.shape[0], -1).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        prune_checkpoints(args)
                        accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                if args.val_data_file and global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        val_dataset = Stage1Dataset(args.val_data_file, args.resolution, args.dataset_root, deterministic=True)
                        log_validation(
                            unwrap(transformer),
                            accelerator,
                            weight_dtype,
                            args,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                            val_dataset,
                            args.resolution,
                            3 * args.resolution,
                            global_step,
                        )
                    accelerator.wait_for_everyone()
                    transformer.train()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        layers = get_peft_model_state_dict(transformer)
        if args.train_norm_layers:
            layers.update({
                f"transformer.{name}": param
                for name, param in transformer.named_parameters()
                if any(key in name for key in NORM_LAYER_PREFIXES)
            })
        FluxFillPipeline.save_lora_weights(args.output_dir, transformer_lora_layers=layers)
        accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
