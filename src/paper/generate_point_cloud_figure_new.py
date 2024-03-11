import sys
sys.path.append('/root/autodl-tmp/pixelsplat')

from pathlib import Path

import hydra
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points

import numpy as np
from PIL import Image
import json
import torchvision.transforms as transforms
from jaxtyping import Float, Int64
import time

SCENES = (
    # scene, [context 1, context 2], far plane
    # ("test", [785, 795], 15, [0]),
    ("1825_1865", [1825, 1865], 15, [0, 30, 60, 90, 120, 150]),
    ("124_128", [124, 128], 15, [0, 30, 60, 90, 120, 150]),
    ("512_522", [512, 522], 15, [0, 30, 60, 90, 120, 150]),
)

FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 8
LINE_WIDTH = 2
LINE_COLOR = [0, 0, 0]
POINT_DENSITY = 0.5

scene_path = "/root/autodl-tmp/SplaTAM/data/Replica/room0/results/"
extrinsics_path = "/root/autodl-tmp/SplaTAM/data/Replica/room0/traj.txt"
intrinsics_path = "/root/autodl-tmp/SplaTAM/data/Replica/cam_params.json"

def read_intrinsics_from_json_tensor(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    
    camera_params = data['camera']
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    w = camera_params['w']
    h = camera_params['h']

    
    normalized_cx = cx /fx
    normalized_cy = cy /fy
    normalized_fx = fx /fx
    normalized_fy = fy /fy

    intrinsics = torch.tensor([[normalized_fx, 0, normalized_cx],
                            [0, normalized_fy, normalized_cy],
                            [0, 0, 1]], dtype=torch.float32)

    intrinsics = torch.stack([intrinsics, intrinsics])  
    device = 'cuda:0'
    intrinsics = intrinsics.to(device)
    return intrinsics

def read_extrinsics(file_path, indices):
    extrinsics = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for index in indices:
        matrix_line = lines[index].strip().split()  
        matrix_line = [float(i) for i in matrix_line]
        matrix = torch.tensor(matrix_line, dtype=torch.float32).view(4, 4)
        extrinsics.append(matrix)
    
    extrinsics = torch.stack(extrinsics, dim=0)
    extrinsics_4d = extrinsics.unsqueeze(0)

    device = 'cuda:0'
    extrinsics_cuda = extrinsics_4d.to(device)
    return extrinsics_cuda



def load_images(indices):
    device = 'cuda:0'
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    images = []
    image_paths = [f"{scene_path}frame{index:06}.jpg" for index in indices]

    for path in image_paths:
        image = Image.open(path)  
        image_tensor = preprocess(image).unsqueeze(0)  
        images.append(image_tensor)

    batch_tensor = torch.cat(images, dim=0)
    batch_tensor_5d = batch_tensor.unsqueeze(0)
    
    batch_tensor_5d = batch_tensor_5d.to(device)
    
    return batch_tensor_5d







@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)

def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval()

    
    for idx, (scene, context_indices, far, angles) in enumerate(SCENES):
        # example = {
        #             "context": {
        #                 "extrinsics": extrinsics[context_indices],
        #                 "intrinsics": intrinsics[context_indices],
        #                 "image": context_images,
        #                 "near": self.get_bound("near", len(context_indices)) / scale,
        #                 "far": self.get_bound("far", len(context_indices)) / scale,
        #                 "index": context_indices,
        #             },
        #             "target": {
        #                 "extrinsics": extrinsics[target_indices],
        #                 "intrinsics": intrinsics[target_indices],
        #                 "image": target_images,
        #                 "near": self.get_bound("near", len(target_indices)) / scale,
        #                 "far": self.get_bound("far", len(target_indices)) / scale,
        #                 "index": target_indices,
        #             },
        #             "scene": scene,
        #         }
        start_time = time.time()
        example = {"context": {}}
        device = 'cuda:0'
        intrinsics = read_intrinsics_from_json_tensor(intrinsics_path)  # Example intrinsics for simplicity
        

        example["context"]["extrinsics"] = read_extrinsics(extrinsics_path, context_indices)
        example["context"]["intrinsics"] = intrinsics.unsqueeze(0)
        example["context"]["image"] = load_images(context_indices)
        # Assuming near and far values are predefined or calculated elsewhere
        example["context"]["near"] = torch.tensor([[0.0598, 0.0598]]).to(device)  # Example values
        example["context"]["far"] = torch.tensor([[597.6885, 597.6885]]).to(device)  # Example values
        example["context"]["index"] = torch.tensor([context_indices]).to(device)

        print(example["context"])
        print("___________")
        print(example["context"]["image"].shape)







        # Generate the Gaussians.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"], False, visualization_dump=visualization_dump
        )


        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        # Transform means into camera space.
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        )
        means = homogenize_points(means)
        w2c = example["context"]["extrinsics"].inverse()[0]
        means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]

        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        mask = mask & (means[..., 2] < far)
        end_time = time.time()
        print("time", end_time-start_time)

        # def trim(element):
        #     element = rearrange(
        #         element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
        #     )
        #     return element[mask][None]

        # for angle in angles:
        #     # Define the pose we render from.
        #     pose = torch.eye(4, dtype=torch.float32, device=device)
        #     rotation = R.from_euler("xyz", [-15, angle - 90, 0], True).as_matrix()
        #     pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        #     translation = torch.eye(4, dtype=torch.float32, device=device)
        #     # visual balance, 0.5x pyramid/frustum volume
        #     translation[2, 3] = far * (0.5 ** (1 / 3))
        #     pose = translation @ pose

        #     ones = torch.ones((1,), dtype=torch.float32, device=device)
        #     render_args = {
        #         "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
        #         "width": ones * far * 2,
        #         "height": ones * far * 2,
        #         "near": ones * 0,
        #         "far": ones * far,
        #         "image_shape": (1024, 1024),
        #         "background_color": torch.zeros(
        #             (1, 3), dtype=torch.float32, device=device
        #         ),
        #         "gaussian_means": trim(gaussians.means),
        #         "gaussian_covariances": trim(gaussians.covariances),
        #         "gaussian_sh_coefficients": trim(gaussians.harmonics),
        #         "gaussian_opacities": trim(gaussians.opacities),
        #     }

        #     # Render alpha (opacity).
        #     dump = {}
        #     alpha_args = {
        #         **render_args,
        #         "gaussian_sh_coefficients": torch.ones_like(
        #             render_args["gaussian_sh_coefficients"][..., :1]
        #         ),
        #         "use_sh": False,
        #     }
        #     alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

        #     # Render (premultiplied) color.
        #     color = render_cuda_orthographic(**render_args)[0]

        #     # Render depths. Without modifying the renderer, we can only render
        #     # premultiplied depth, then hackily transform it into straight alpha depth,
        #     # which is needed for sorting.
        #     depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
        #     depth = depth.norm(dim=-1)
        #     depth_args = {
        #         **render_args,
        #         "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
        #         "use_sh": False,
        #     }
        #     depth_premultiplied = render_cuda_orthographic(**depth_args)
        #     depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

        #     # Save the rendering for later depth-based alpha compositing.
        #     layers = [(color, alpha, depth)]

        #     # Figure out the intrinsics from the FOV.
        #     fx = 0.5 / (0.5 * dump["fov_x"]).tan()
        #     fy = 0.5 / (0.5 * dump["fov_y"]).tan()
        #     dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        #     dump_intrinsics[0, 0] = fx
        #     dump_intrinsics[1, 1] = fy
        #     dump_intrinsics[:2, 2] = 0.5

        #     # Compute frustum corners for the context views.
        #     frustum_corners = unproject_frustum_corners(
        #         example["context"]["extrinsics"][0],
        #         example["context"]["intrinsics"][0],
        #         torch.ones((2,), dtype=torch.float32, device=device) * far / 8,
        #     )
        #     camera_origins = example["context"]["extrinsics"][0, :, :3, 3]

        #     # Generate the 3D lines that have to be computed.
        #     lines = []
        #     for corners, origin in zip(frustum_corners, camera_origins):
        #         for i in range(4):
        #             lines.append((corners[i], corners[i - 1]))
        #             lines.append((corners[i], origin))

        #     # Generate an alpha compositing layer for each line.
        #     for a, b in lines:
        #         # Start with the point whose depth is further from the camera.
        #         a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
        #         b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
        #         start = a if (a_depth > b_depth).all() else b
        #         end = b if (a_depth > b_depth).all() else a

        #         # Create the alpha mask (this one is clean).
        #         start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
        #         end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
        #         alpha = draw_lines(
        #             torch.zeros_like(color),
        #             start_2d[None],
        #             end_2d[None],
        #             (1, 1, 1),
        #             LINE_WIDTH,
        #             x_range=(0, 1),
        #             y_range=(0, 1),
        #         )

        #         # Create the color.
        #         lc = torch.tensor(LINE_COLOR, dtype=torch.float32, device=device)
        #         color = draw_lines(
        #             torch.zeros_like(color),
        #             start_2d[None],
        #             end_2d[None],
        #             lc,
        #             LINE_WIDTH,
        #             x_range=(0, 1),
        #             y_range=(0, 1),
        #         )

        #         # Create the depth. We just individually render points.
        #         wh = torch.tensor((w, h), dtype=torch.float32, device=device)
        #         delta = (wh * (start_2d - end_2d)).norm()
        #         num_points = delta / POINT_DENSITY
        #         t = torch.linspace(0, 1, int(num_points) + 1, device=device)
        #         xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
        #         depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
        #         depth = repeat(depth, "p -> p c", c=3)
        #         xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
        #         depth = draw_points(
        #             torch.ones_like(color) * 1e10,
        #             xy,
        #             depth,
        #             LINE_WIDTH,  # makes it 2x as wide as line
        #             x_range=(0, 1),
        #             y_range=(0, 1),
        #         )

        #         layers.append((color, alpha, depth))

        #     # Do the alpha compositing.
        #     canvas = torch.ones_like(color)
        #     colors = torch.stack([x for x, _, _ in layers])
        #     alphas = torch.stack([x for _, x, _ in layers])
        #     depths = torch.stack([x for _, _, x in layers])
        #     index = depths.argsort(dim=0)
        #     colors = colors.gather(index=index, dim=0)
        #     alphas = alphas.gather(index=index, dim=0)
        #     t = (1 - alphas).cumprod(dim=0)
        #     t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
        #     image = (t * colors).sum(dim=0)
        #     total_alpha = (t * alphas).sum(dim=0)
        #     image = total_alpha * image + (1 - total_alpha) * canvas

        #     base = Path(f"point_clouds/{idx:0>6}_{scene}")
        #     save_image(image, f"{base}_angle_{angle:0>3}.png")

        #     # Render depth.
        #     *_, h, w = example["context"]["image"].shape
        #     rendered = decoder.forward(
        #         gaussians,
        #         example["context"]["extrinsics"],
        #         example["context"]["intrinsics"],
        #         example["context"]["near"],
        #         example["context"]["far"],
        #         (h, w),
        #         "depth",
        #     )

        #     export_ply(
        #         example["context"]["extrinsics"][0, 0],
        #         trim(gaussians.means)[0],
        #         trim(visualization_dump["scales"])[0],
        #         trim(visualization_dump["rotations"])[0],
        #         trim(gaussians.harmonics)[0],
        #         trim(gaussians.opacities)[0],
        #         base / "gaussians.ply",
        #     )

        #     result = rendered.depth
        #     depth_near = result[result > 0].quantile(0.01).log()
        #     depth_far = result.quantile(0.99).log()
        #     result = result.log()
        #     result = 1 - (result - depth_near) / (depth_far - depth_near)
        #     result = apply_color_map_to_image(result, "turbo")
        #     save_image(result[0, 0], f"{base}_depth_0.png")
        #     save_image(result[0, 1], f"{base}_depth_1.png")
        #     a = 1
        # a = 1
    a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
