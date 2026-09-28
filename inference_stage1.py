import argparse
import os

import numpy as np
import torch

from utils import decode_coc, get_stage_dir, load_pipeline, load_prompt_embeds, prepare_stage1_inputs, unpad_and_resize


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: predict the CoC and disparity maps of an image.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument(
        "--anybokeh_model_name_or_path",
        type=str,
        default="itsmag11/AnyBokeh",
        help="Hub repo id or local folder with the stage1/ and stage2/ checkpoints.",
    )
    parser.add_argument("--coc_save_path", type=str, required=True)
    parser.add_argument("--disp_save_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def predict_coc_and_disparity(pipe, prompt_embeds_dir, image_path, resolution, num_inference_steps, seed):
    prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(prompt_embeds_dir)
    image, mask, info = prepare_stage1_inputs(image_path, resolution)
    output = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=image,
        mask_image=mask,
        height=resolution,
        width=3 * resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=30.0,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    output = np.array(output)

    r = resolution
    coc = unpad_and_resize(decode_coc(output[:, r:2 * r]), info) / info["scale"]
    disp = unpad_and_resize(output[:, 2 * r:].astype(np.float32) / 255.0, info)
    return coc, disp


def main(args):
    stage_dir = get_stage_dir(args.anybokeh_model_name_or_path, "stage1")
    pipe = load_pipeline(args.pretrained_model_name_or_path)
    pipe.load_lora_weights(stage_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.vae.enable_tiling()

    coc, disp = predict_coc_and_disparity(
        pipe, stage_dir, args.image_path, args.resolution, args.num_inference_steps, args.seed
    )

    for path, x in [(args.coc_save_path, coc), (args.disp_save_path, disp)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, x)


if __name__ == "__main__":
    args = parse_args()
    main(args)
