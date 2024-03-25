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
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim


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
import time

SCENES = (
    # scene, [context 1, context 2], far plane
    # ("test", [785, 795], 15, [0]),
    ("1825_1865", [1825, 1865], 15, [0, 30, 60, 90, 120, 150]),
    # ("124_128", [124, 128], 15, [0, 30, 60, 90, 120, 150]),
    # ("512_522", [512, 522], 15, [0, 30, 60, 90, 120, 150]),
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


class PointCloudGenerator():
    def __init__(self, cfg):
        self.cfg = load_typed_root_config(cfg)
        set_cfg(cfg)
        torch.manual_seed(cfg.seed)
        self.device = torch.device("cuda:0")
        self.checkpoint_path = update_checkpoint_path(self.cfg.checkpointing.load, self.cfg.wandb)
        self.encoder, self.encoder_visualizer = get_encoder(self.cfg.model.encoder)
        self.decoder = get_decoder(self.cfg.model.decoder, self.cfg.dataset)
        self.model_wrapper = self.load_model()
        self.model_wrapper.eval()
        self.intrinsics = self.read_intrinsics_from_json_tensor(intrinsics_path)
        self.index = 0
        self.extrinsics = self.read_extrinsics(extrinsics_path)
        self.far = 10.0

    def load_model(self):
        model_wrapper = ModelWrapper.load_from_checkpoint(
            self.checkpoint_path,
            optimizer_cfg=self.cfg.optimizer,
            test_cfg=self.cfg.test,
            train_cfg=self.cfg.train,
            encoder=self.encoder,
            encoder_visualizer=self.encoder_visualizer,
            decoder=self.decoder,
            losses=[],
            step_tracker=None,
        )
        return model_wrapper
    
    def read_intrinsics_from_json_tensor(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        fx = 600.0 / 1200
        fy = 600.0 / 680
        cx = 0.5
        cy = 0.5

        intrinsics = torch.tensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]], dtype=torch.float32)

        intrinsics = torch.stack([intrinsics, intrinsics])  
        intrinsics = intrinsics.to(self.device)
        return intrinsics

    def read_extrinsics(self, file_path):
        extrinsics = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            matrix_line = line.strip().split()  
            matrix_line = [float(i) for i in matrix_line]
            matrix = torch.tensor(matrix_line, dtype=torch.float32).view(4, 4)
            extrinsics.append(matrix)
        
        extrinsics = torch.stack(extrinsics, dim=0)
        extrinsics = extrinsics.unsqueeze(0)
        extrinsics = extrinsics.to(self.device)

        return extrinsics


    def load_images(self, time_idx):
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        images = []
        image_paths = [f"{scene_path}frame{index:06}.jpg" for index in [time_idx-1, time_idx]]

        for path in image_paths:
            image = Image.open(path)  
            image_tensor = preprocess(image).unsqueeze(0)  
            images.append(image_tensor)

        batch_tensor = torch.cat(images, dim=0)
        batch_tensor = batch_tensor.unsqueeze(0)
        
        return batch_tensor

    def generate_gaussians(self, time_idx, last_w2c, curr_w2c, last_img, curr_img, intrinsics, render):

        start_time = time.time()
        example = {"context": {}}

        extrinsics = torch.stack([curr_w2c.inverse(), last_w2c.inverse()], dim=0).unsqueeze(0).to(self.device)
        example["context"]["extrinsics"] = extrinsics
        example["context"]["image"] = torch.stack([curr_img, last_img], dim=0).unsqueeze(0).to(self.device)
        example["context"]["near"] = torch.tensor([[0.05, 0.05]]).to(self.device)  # Example values
        example["context"]["far"] = torch.tensor([[10.0, 10.0]]).to(self.device)  # Example values
        example["context"]["index"] = torch.tensor([time_idx, time_idx-4]).to(self.device)
        example["context"]["intrinsics"] = torch.tensor([[[
          [0.5000, 0.0000, 0.5000],
          [0.0000, 0.8824, 0.5000],
          [0.0000, 0.0000, 1.0000]],

         [[0.5000, 0.0000, 0.5000],
          [0.0000, 0.8824, 0.5000],
          [0.0000, 0.0000, 1.0000]]]], device='cuda:0')
        # example["context"]["image"] = self.load_images(1).to(self.device)
        # example["context"]["intrinsics"] = torch.stack([intrinsics, intrinsics], dim=0).unsqueeze(0).to(self.device)
        # example["context"]["extrinsics"] = torch.tensor([[[
        #   [-3.2057e-01,  4.4806e-01, -8.3455e-01,  3.4530e+00],
        #   [ 9.4722e-01,  1.5164e-01, -2.8244e-01,  4.5461e-01],
        #   [ 1.0790e-16, -8.8105e-01, -4.7302e-01,  5.9363e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[-3.1393e-01,  4.5307e-01, -8.3437e-01,  3.4572e+00],
        #   [ 9.4945e-01,  1.4981e-01, -2.7588e-01,  4.6971e-01],
        #   [ 1.0762e-16, -8.7880e-01, -4.7719e-01,  5.9427e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]], device='cuda:0')

        visualization_dump = {}
        gaussians = self.encoder.forward(
            example["context"], True, visualization_dump=visualization_dump
        )

        # # Figure out which Gaussians to mask off/throw away.
        # _, _, _, h, w = example["context"]["image"].shape

        # # Transform means into camera space.
        # means = rearrange(
        #     gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        # )
        # means = homogenize_points(means)
        # w2c = example["context"]["extrinsics"].inverse()[0]

        # means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]

        # # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # # borders, since they're generally of lower quality.
        # mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        # mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # # Then, drop Gaussians that are really far away.
        # mask = mask & (means[..., 2] < self.far)


        # def trim(element):
        #     element = rearrange(
        #         element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
        #     )
        #     return element[mask][None]

        # gaussians.means =  trim(gaussians.means)
        # gaussians.covariances = trim(gaussians.covariances)
        # gaussians.harmonics = trim(gaussians.harmonics)
        # gaussians.opacities = trim(gaussians.opacities)

        op_mask = gaussians.opacities < 0.15
        gaussians.means = gaussians.means[~op_mask].unsqueeze(0)
        gaussians.covariances = gaussians.covariances[~op_mask].unsqueeze(0)
        gaussians.harmonics = gaussians.harmonics[~op_mask].unsqueeze(0)
        gaussians.opacities = gaussians.opacities[~op_mask].unsqueeze(0)

        print(gaussians.means.shape)
        # print(gaussians.covariances.shape)
        # print(gaussians.harmonics.shape)
        # print(gaussians.opacities.shape)
        # print(gaussians.covariances)
        # print(gaussians.harmonics)
        end_time = time.time()
        print("time", end_time-start_time)

        if render:
            *_, h, w = example["context"]["image"].shape
            rendered = self.decoder.forward(
                gaussians,
                example["context"]["extrinsics"],
                example["context"]["intrinsics"],
                example["context"]["near"],
                example["context"]["far"],
                (h, w),
                "depth",
            )

            # print(example["context"]["extrinsics"])
            # time.sleep(10)

            target_gt = example["context"]["image"]

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(rendered.color, "b v c h w -> (b v) c h w"),
            )
            print("train/psnr_probabilistic", psnr_probabilistic.mean())

            for i in range(rendered.color.size(1)):
                save_image(rendered.color[0, i], f'spimg_{time_idx}_{i}.png')

            for i in range(target_gt.size(1)):
                save_image(target_gt[0, i], f'spgt_{time_idx}_{i}.png')

            return gaussians, psnr_probabilistic.mean()

        return gaussians


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)

def main(cfg):
    with torch.no_grad():
        point_cloud_generator = PointCloudGenerator(cfg)
        gaussians = point_cloud_generator.generate_gaussians()
        print(gaussians)

if __name__ == "__main__":
    main()

