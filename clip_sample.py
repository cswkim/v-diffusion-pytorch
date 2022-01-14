#!/usr/bin/env python3

"""CLIP guided sampling from a diffusion model."""

import argparse
from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = [prompt]
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def main(prompts: list[str] = [], image_prompts: list[str] = [], batch_size: int = 1,
         checkpoint: str = None, clip_guidance_scale: float = 500., cutn: int = 16,
         cut_pow: float = 1., device_name: str = None, eta: float = 1., init_image: str = None,
         model_name: str = 'cc12m_1', num_samples: int = 1, seed: int = 0,
         size: tuple[int, int] = (256, 256), start_timestep: float = 0.9, num_steps: int = 1000):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = get_model(model_name)()
    _, side_y, side_x = model.shape
    if size:
        side_x, side_y = size

    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{model_name}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn, cut_pow)

    if init_image:
        init = Image.open(utils.fetch(init_image)).convert('RGB')
        init = resize_and_center_crop(init, (side_x, side_y))
        init = utils.from_pil_image(init).to(device)[None].repeat([num_samples, 1, 1, 1])

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size),
                        transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img)[None].to(device))
        embeds = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embeds)
        weights.extend([weight / cutn] * cutn)

    if not target_embeds:
        raise RuntimeError('At least one text or image prompt must be specified.')
    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    clip_embed = F.normalize(target_embeds.mul(weights[:, None]).sum(0, keepdim=True), dim=-1)
    clip_embed = clip_embed.repeat([num_samples, 1])

    torch.manual_seed(seed)

    def cond_fn(x, t, pred, clip_embed):
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * clip_guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, steps, clip_embed):
        if hasattr(model, 'clip_model'):
            extra_args = {'clip_embed': clip_embed}
            cond_fn_ = cond_fn
        else:
            extra_args = {}
            cond_fn_ = partial(cond_fn, clip_embed=clip_embed)
        if not clip_guidance_scale:
            return sampling.sample(model, x, steps, eta, extra_args)
        return sampling.cond_sample(model, x, steps, eta, extra_args, cond_fn_)

    def run_all(n, batch_size):
        x = torch.randn([n, 3, side_y, side_x], device=device)
        t = torch.linspace(1, 0, num_steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        if init_image:
            steps = steps[steps < start_timestep]
            alpha, sigma = utils.t_to_alpha_sigma(steps[0])
            x = init * alpha + x * sigma
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i:i+cur_batch_size], steps, clip_embed[i:i+cur_batch_size])
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'obj_diff_pytorch_{i + j:05}.png')

    try:
        run_all(num_samples, batch_size)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
