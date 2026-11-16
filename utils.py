import os
import random
import shutil

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter

# ==============================================================================
# Dataset files
# ==============================================================================

def load_float_map(path):
    """Load a `.npy` map as a float32 array. A trailing channel axis is dropped."""
    array = np.load(path, allow_pickle=True).astype(np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array


def coc_path_from_rgb(rgb_path):
    """`.../rgb/name.jpeg` -> `.../coc/name.npy`."""
    stem, _ = os.path.splitext(rgb_path.replace("/rgb/", "/coc/"))
    return stem + ".npy"


def disparity_path_from_depth(depth_path):
    """`.../depth/name.exr` -> `.../disparity/name.npy`."""
    stem, _ = os.path.splitext(depth_path.replace("/depth/", "/disparity/"))
    return stem + ".npy"

# ==============================================================================
# Circle of confusion (CoC)
# ==============================================================================


def encode_stage1_coc(coc):
    """Signed-log encoding used to train Stage 1. Inverse of `decode_stage1_coc` (max CoC = 50 px)."""
    coc = np.asarray(coc, dtype=np.float32)
    if coc.ndim == 3:
        coc = coc[..., 0]
    signed = np.sign(coc) * (np.log1p(np.abs(coc)) / np.log1p(50.0))
    x = np.clip((signed + 1.0) / 2.0, 0.0, 1.0)
    return Image.fromarray((x * 255).astype(np.uint8)).convert("RGB")

def decode_stage1_coc(pixels):
    x = 2.0 * (pixels.astype(np.float32) / 255.0) - 1.0
    x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.sign(x) * np.expm1(np.abs(x) * np.log1p(50.0))

def encode_stage2_coc(coc, resolution):
    x = np.tanh(np.asarray(coc, dtype=np.float32) / (resolution * 0.1))
    x = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    return Image.fromarray((x * 255).astype(np.uint8)).convert("RGB")


def compute_target_coc(coc, disparity, focus_xy, bokeh_scale):
    """Refocus the CoC map at `focus_xy` and scale it by `bokeh_scale` (N_src / N_tgt).

    CoC is linear in disparity, C = K * (d_focus - d). The source focus plane is taken from the
    sharpest pixels and K from the blurriest ones.
    """
    coc = np.asarray(coc, dtype=np.float32)
    disparity = np.asarray(disparity, dtype=np.float32)
    if coc.ndim == 3:
        coc = coc[..., 0]
    if disparity.ndim == 3:
        disparity = disparity[..., 0]

    abs_coc = np.abs(coc)
    sharp = abs_coc <= np.quantile(abs_coc, 0.01)
    source_focus = np.median(disparity[sharp]).astype(np.float32)

    delta = source_focus - disparity
    valid = (abs_coc >= np.quantile(abs_coc, 0.95)) & (np.abs(delta) > 1e-3)
    k = np.abs(np.median(coc[valid] / delta[valid]).astype(np.float32))

    target_focus = disparity[focus_xy[1], focus_xy[0]]
    return (k * np.float32(bokeh_scale) * (target_focus - disparity)).astype(np.float32)

# ==============================================================================
# Resizing, padding, and tiling
# ==============================================================================

def resize_map(x, size):
    """Bilinearly resize an (H, W) or (H, W, C) float map to size=(W, H)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    t = t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
    t = TF.resize(t, [size[1], size[0]], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    t = t.squeeze(0) if x.ndim == 2 else t.permute(1, 2, 0)
    return t.numpy()


def resize_and_pad(image, size):
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = TF.resize(image, (new_h, new_w))

    # reflect padding needs each side to be smaller than the image
    def split(total, dim):
        max_side = max(dim - 1, 0)
        return max(min(max(total // 2, total - max_side), max_side), 0)

    pad_w, pad_h = size - new_w, size - new_h
    left, top = split(pad_w, new_w), split(pad_h, new_h)
    padding = [left, top, pad_w - left, pad_h - top]
    if pad_w > 2 * max(new_w - 1, 0) or pad_h > 2 * max(new_h - 1, 0):
        mode = "edge"
    else:
        mode = "reflect"
    if max(padding) > 0:
        image = TF.pad(image, padding, padding_mode=mode)

    info = {"orig_size": (w, h), "scale": scale, "offset": (left, top), "resized_size": (new_w, new_h)}
    return image, info


def unpad_and_resize(x, info):
    left, top = info["offset"]
    w, h = info["resized_size"]
    return resize_map(x[top:top + h, left:left + w], info["orig_size"])


def get_tile_offsets(canvas_size, tile_size):
    """Top-left corners of overlapping tiles (20% overlap) covering a canvas of size=(W, H)."""
    stride = round(tile_size * 0.8)

    def starts(length):
        if length <= tile_size:
            return [0]
        return list(range(0, length - tile_size, stride)) + [length - tile_size]

    w, h = canvas_size
    return [(x, y) for y in starts(h) for x in starts(w)]


def blend_tiles(tiles, offsets, canvas_size, tile_size):
    """Merge overlapping tiles with a 2D Hanning window."""
    w, h = canvas_size
    window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)[..., None]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.zeros((h, w, 1), dtype=np.float32)
    for tile, (x, y) in zip(tiles, offsets):
        canvas[y:y + tile_size, x:x + tile_size] += tile.astype(np.float32) * window
        weights[y:y + tile_size, x:x + tile_size] += window
    return np.clip(canvas / np.maximum(weights, 1e-6), 0, 255).astype(np.uint8)


def upsample_to_min_size(image, coc, target_coc, min_size):
    """Upsample so that the short edge is at least `min_size`. CoC values are in pixels, so they are scaled too."""
    w, h = image.size
    if min(w, h) >= min_size:
        return image, coc, target_coc, 1.0

    scale = min_size / min(w, h)
    new_w, new_h = max(round(w * scale), min_size), max(round(h * scale), min_size)
    image = TF.resize(image, (new_h, new_w))
    coc = resize_map(coc, (new_w, new_h)) * scale
    target_coc = resize_map(target_coc, (new_w, new_h)) * scale
    return image, coc, target_coc, scale

# ==============================================================================
# Inference arguments
# ==============================================================================

def add_bokeh_args(parser):
    parser.add_argument("--focus_x", type=int, required=True, help="Column of the focus point, in input image pixels.")
    parser.add_argument("--focus_y", type=int, required=True, help="Row of the focus point, in input image pixels.")
    parser.add_argument(
        "--bokeh_scale",
        type=float,
        default=None,
        help="Scale of the bokeh size relative to the input (>1: stronger, <1: weaker). "
        "Alternative to --source_aperture/--target_aperture.",
    )
    parser.add_argument("--source_aperture", type=float, default=None, help="f-number of the input image.")
    parser.add_argument("--target_aperture", type=float, default=None, help="f-number to render.")


def get_bokeh_scale(parser, args):
    """Bokeh scale from --bokeh_scale, or N_src / N_tgt from the f-numbers, since CoC is inversely proportional to N."""
    has_apertures = args.source_aperture is not None or args.target_aperture is not None
    if args.bokeh_scale is not None:
        if has_apertures:
            parser.error("use either --bokeh_scale or --source_aperture/--target_aperture, not both")
        scale = args.bokeh_scale
    elif args.source_aperture is not None and args.target_aperture is not None:
        if args.source_aperture <= 0 or args.target_aperture <= 0:
            parser.error("--source_aperture and --target_aperture must be positive")
        scale = args.source_aperture / args.target_aperture
    else:
        parser.error("specify --bokeh_scale, or both --source_aperture and --target_aperture")
    if scale <= 0:
        parser.error("--bokeh_scale must be positive")
    return scale

# ==============================================================================
# Inference model loading
# ==============================================================================

def get_stage_dir(model_name_or_path, stage):
    """Folder of `{stage}-*` checkpoint files. A repo id is downloaded for that prefix."""
    if not os.path.isdir(model_name_or_path):
        model_name_or_path = snapshot_download(model_name_or_path, allow_patterns=f"{stage}-*")
    return model_name_or_path


def lora_weight_name(stage):
    return f"{stage}-pytorch_lora_weights.safetensors"


def load_pipeline(pretrained_model_name_or_path):
    pipe = FluxFillPipeline.from_pretrained(
        pretrained_model_name_or_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def load_prompt_embeds(checkpoint_dir, stage):
    prompt_embeds = torch.load(os.path.join(checkpoint_dir, f"{stage}-prompt_embeds.pt")).to("cuda")
    pooled_prompt_embeds = torch.load(os.path.join(checkpoint_dir, f"{stage}-pooled_prompt_embeds.pt")).to("cuda")
    return prompt_embeds, pooled_prompt_embeds

# ==============================================================================
# Inference canvases
# ==============================================================================

def prepare_stage1_inputs(image_path, resolution):
    """Returns the [image | CoC | disparity] canvas, its inpainting mask and the padding info."""
    image, info = resize_and_pad(load_image(image_path), resolution)
    canvas = Image.new("RGB", (3 * resolution, resolution))
    canvas.paste(image, (0, 0))
    mask = Image.new("L", (3 * resolution, resolution), 255)
    mask.paste(0, (0, 0, resolution, resolution))
    return canvas, mask, info


def prepare_stage2_inputs(image, coc, target_coc, resolution):
    """Returns the 2x2 grid [source CoC | image; target CoC | output] and its inpainting mask."""
    grid = Image.new("RGB", (2 * resolution, 2 * resolution))
    grid.paste(encode_stage2_coc(coc, resolution), (0, 0))
    grid.paste(image, (resolution, 0))
    grid.paste(encode_stage2_coc(target_coc, resolution), (0, resolution))
    mask = Image.new("L", (2 * resolution, 2 * resolution), 0)
    mask.paste(255, (resolution, resolution, 2 * resolution, 2 * resolution))
    return grid, mask

# ==============================================================================
# Stage 1 training
# ==============================================================================

def disparity_to_image(disp):
    """Per-map min-max normalization of a disparity array to an RGB image."""
    disp = np.asarray(disp, dtype=np.float32)
    if disp.ndim == 3:
        disp = disp[..., 0]
    span = float(disp.max() - disp.min())
    disp = np.zeros_like(disp) if span < 1e-8 else (disp - disp.min()) / span
    return Image.fromarray((disp * 255).astype(np.uint8)).convert("RGB")


def fit_long_edge_and_pad(images, size, deterministic=False):
    """Resize the long edge to `size`, then pad to a square.

    Training draws a random pad offset and clamps it so reflect-pad stays valid.
    `deterministic=True` center-pads, which is what validation uses.
    CoC values are not rescaled here; multiply them by the returned scale yourself.
    """
    width, height = images[0].size
    scale = size / max(width, height)
    new_w, new_h = round(width * scale), round(height * scale)
    images = [TF.resize(image, (new_h, new_w)) for image in images]

    pad_w, pad_h = max(0, size - new_w), max(0, size - new_h)

    def clamp_split(offset, total, dim):
        max_side = max(dim - 1, 0)
        offset = min(max(offset, total - max_side), max_side)
        return max(0, offset)

    if deterministic:
        left, top = pad_w // 2, pad_h // 2
    else:
        left = random.randint(0, pad_w) if pad_w > 0 else 0
        top = random.randint(0, pad_h) if pad_h > 0 else 0
    left = clamp_split(left, pad_w, new_w)
    top = clamp_split(top, pad_h, new_h)
    padding = [left, top, pad_w - left, pad_h - top]
    if pad_w > 2 * max(new_w - 1, 0) or pad_h > 2 * max(new_h - 1, 0):
        mode = "edge"
    else:
        mode = "reflect"
    if max(padding) > 0:
        images = [TF.pad(image, padding, padding_mode=mode) for image in images]
    if new_w > size or new_h > size:
        images = [TF.center_crop(image, (size, size)) for image in images]
    return images, scale

# ==============================================================================
# Stage 2 training
# ==============================================================================

def stage2_training_tensors(image, coc, target_image, target_coc, resolution, augment_coc=False, random_flip=True):
    """Random crop. Returns normalized `(pixel_values, masked_image, mask)` for the 2x2 grid.

    `augment_coc` applies the edge perturbations used while pretraining on ground-truth CoC.
    """
    image, coc, target_coc, scale = upsample_to_min_size(image, np.asarray(coc, dtype=np.float32), np.asarray(target_coc, dtype=np.float32), resolution)
    if scale != 1.0:
        target_image = TF.resize(target_image, (image.size[1], image.size[0]))
    if random_flip and random.random() < 0.5:
        image = TF.hflip(image)
        target_image = TF.hflip(target_image)
        coc = np.flip(np.asarray(coc), axis=1).copy()
        target_coc = np.flip(np.asarray(target_coc), axis=1).copy()

    width, height = image.size
    left = random.randint(0, width - resolution)
    top = random.randint(0, height - resolution)
    image = TF.crop(image, top, left, resolution, resolution)
    target_image = TF.crop(target_image, top, left, resolution, resolution)
    coc = np.asarray(coc)[top:top + resolution, left:left + resolution]
    target_coc = np.asarray(target_coc)[top:top + resolution, left:left + resolution]

    coc_image = encode_stage2_coc(coc, resolution)
    target_coc_image = encode_stage2_coc(target_coc, resolution)
    if augment_coc:
        coc_image = _perturb_coc(coc_image)
        target_coc_image = _perturb_coc(target_coc_image)

    grid = Image.new("RGB", (2 * resolution, 2 * resolution))
    grid.paste(coc_image, (0, 0))
    grid.paste(image, (resolution, 0))
    grid.paste(target_coc_image, (0, resolution))
    grid.paste(target_image, (resolution, resolution))

    mask = torch.zeros(1, 2 * resolution, 2 * resolution)
    mask[:, resolution:, resolution:] = 1
    values = TF.to_tensor(grid)
    masked = values * (1 - mask)
    values = values * 2 - 1
    masked = masked * 2 - 1
    return values, masked.unsqueeze(0), mask


def _perturb_coc(image):
    """Elastic warp, morphology, or blur, each applied with probability 1/2."""
    if random.random() > 0.5:
        image = _elastic_coc(image)
    if random.random() > 0.5:
        image = _morph_coc(image)
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    return image


def _elastic_coc(image):
    from scipy.ndimage import gaussian_filter, map_coordinates

    array = np.array(image, dtype=np.float32)
    alpha = random.uniform(1.0, 3.0)
    sigma = random.uniform(4.0, 6.0)
    shape = array.shape[:2]
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    coords = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]
    if array.ndim == 3:
        out = np.stack([map_coordinates(array[:, :, c], coords, order=1, mode="reflect") for c in range(array.shape[2])], axis=-1)
    else:
        out = map_coordinates(array, coords, order=1, mode="reflect")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _morph_coc(image):
    import cv2

    array = np.array(image)
    size = random.choice((3, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    out = cv2.dilate(array, kernel, iterations=1) if random.random() > 0.5 else cv2.erode(array, kernel, iterations=1)
    return Image.fromarray(out)

# ==============================================================================
# Training functions
# ==============================================================================

def encode_prompt(args, weight_dtype, device):
    pipe = FluxFillPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        args.global_text_prompt, prompt_2=None, device=device, max_sequence_length=512
    )
    del pipe
    return prompt_embeds[:1].to(device), pooled_prompt_embeds[:1].to(device), text_ids.to(device)


def encode_vae(pixels, vae, weight_dtype):
    latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(weight_dtype)


def pack_mask(mask, latents, vae_scale_factor):
    mask = mask[:, 0]
    mask = mask.view(latents.shape[0], latents.shape[2], vae_scale_factor, latents.shape[3], vae_scale_factor)
    mask = mask.permute(0, 2, 4, 1, 3)
    return mask.reshape(latents.shape[0], vae_scale_factor * vae_scale_factor, latents.shape[2], latents.shape[3])


def prune_checkpoints(args):
    from accelerate.logging import get_logger

    logger = get_logger(__name__)
    if args.checkpoints_total_limit is None:
        return
    names = sorted(
        [name for name in os.listdir(args.output_dir) if name.startswith("checkpoint-")],
        key=lambda name: int(name.split("-")[1]),
    )
    extra = names[: max(0, len(names) - args.checkpoints_total_limit + 1)]
    for name in extra:
        logger.info(f"Removing {name}")
        shutil.rmtree(os.path.join(args.output_dir, name))
