import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers.utils import load_image
from PIL import Image

from utils import (
    add_bokeh_args,
    blend_tiles,
    compute_target_coc,
    get_bokeh_scale,
    get_stage_dir,
    get_tile_offsets,
    load_pipeline,
    load_prompt_embeds,
    prepare_stage2_inputs,
    resize_map,
    upsample_to_min_size,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: re-render an image with a new focus point and aperture.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--coc_path", type=str, required=True, help="CoC map predicted by stage 1.")
    parser.add_argument("--disp_path", type=str, required=True, help="Disparity map predicted by stage 1.")
    add_bokeh_args(parser)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument(
        "--anybokeh_model_name_or_path",
        type=str,
        default="itsmag11/AnyBokeh",
        help="Hub repo id or local folder with the stage1/ and stage2/ checkpoints.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.bokeh_scale = get_bokeh_scale(parser, args)
    return args


@torch.no_grad()
def render_bokeh(
    pipe, prompt_embeds_dir, image, coc, disp, focus_xy, bokeh_scale, resolution, num_inference_steps, seed
):
    w, h = image.size
    if not (0 <= focus_xy[0] < w and 0 <= focus_xy[1] < h):
        raise ValueError(f"Focus point {tuple(focus_xy)} is outside the {w}x{h} image.")

    target_coc = compute_target_coc(coc, disp, focus_xy, bokeh_scale)
    prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(prompt_embeds_dir)

    # The model works on resolution x resolution tiles, so large images are split and blended back.
    image, coc, target_coc, scale = upsample_to_min_size(image, coc, target_coc, resolution)
    offsets = get_tile_offsets(image.size, resolution)
    tiles = []
    for i, (x, y) in enumerate(offsets):
        grid, mask = prepare_stage2_inputs(
            TF.crop(image, y, x, resolution, resolution),
            coc[y:y + resolution, x:x + resolution],
            target_coc[y:y + resolution, x:x + resolution],
            resolution,
        )
        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=grid,
            mask_image=mask,
            height=2 * resolution,
            width=2 * resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=30.0,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed + i),
        ).images[0]
        tiles.append(np.array(output)[resolution:, resolution:])

    result = blend_tiles(tiles, offsets, image.size, resolution)
    if scale != 1.0:
        result = np.clip(resize_map(result, (w, h)), 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def main(args):
    stage_dir = get_stage_dir(args.anybokeh_model_name_or_path, "stage2")
    pipe = load_pipeline(args.pretrained_model_name_or_path)
    pipe.load_lora_weights(stage_dir, weight_name="pytorch_lora_weights.safetensors")

    result = render_bokeh(
        pipe,
        stage_dir,
        load_image(args.image_path),
        np.load(args.coc_path),
        np.load(args.disp_path),
        (args.focus_x, args.focus_y),
        args.bokeh_scale,
        args.resolution,
        args.num_inference_steps,
        args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result.save(args.output_path, quality=95)


if __name__ == "__main__":
    args = parse_args()
    main(args)
