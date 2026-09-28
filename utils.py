import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
from PIL import Image


# ==============================================================================
# Command-line arguments
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
# Model loading
# ==============================================================================

def get_stage_dir(model_name_or_path, stage):
    """Local folder with the LoRA weights and prompt embeddings of `stage`, downloaded from the Hub if needed."""
    if not os.path.isdir(model_name_or_path):
        model_name_or_path = snapshot_download(model_name_or_path, allow_patterns=f"{stage}/*")
    return os.path.join(model_name_or_path, stage)


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


def load_prompt_embeds(prompt_embeds_dir):
    prompt_embeds = torch.load(os.path.join(prompt_embeds_dir, "prompt_embeds.pt")).to("cuda")
    pooled_prompt_embeds = torch.load(os.path.join(prompt_embeds_dir, "pooled_prompt_embeds.pt")).to("cuda")
    return prompt_embeds, pooled_prompt_embeds


# ==============================================================================
# Image resizing, padding and tiling
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


# ==============================================================================
# Circle of confusion (CoC)
# ==============================================================================

def decode_coc(pixels):
    x = 2.0 * (pixels.astype(np.float32) / 255.0) - 1.0
    x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.sign(x) * np.expm1(np.abs(x) * np.log1p(50.0))


def encode_coc(coc, resolution):
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
# Stage input layouts
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
    grid.paste(encode_coc(coc, resolution), (0, 0))
    grid.paste(image, (resolution, 0))
    grid.paste(encode_coc(target_coc, resolution), (0, resolution))
    mask = Image.new("L", (2 * resolution, 2 * resolution), 0)
    mask.paste(255, (resolution, resolution, 2 * resolution, 2 * resolution))
    return grid, mask
