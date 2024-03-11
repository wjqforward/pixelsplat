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


class PointCloudGenerator:
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

    def read_extrinsics(self, file_path, indices):
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



    def load_images(self, indices):
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

    def generate_gaussians(self):
        for idx, (scene, context_indices, far, angles) in enumerate(SCENES):
            start_time = time.time()
            example = {"context": {}}
            device = 'cuda:0'
            intrinsics = self.read_intrinsics_from_json_tensor(intrinsics_path)  # Example intrinsics for simplicity
            
            example["context"]["extrinsics"] = self.read_extrinsics(extrinsics_path, context_indices)
            example["context"]["intrinsics"] = intrinsics.unsqueeze(0)
            example["context"]["image"] = self.load_images(context_indices)
            # Assuming near and far values are predefined or calculated elsewhere
            example["context"]["near"] = torch.tensor([[0.0598, 0.0598]]).to(device)  # Example values
            example["context"]["far"] = torch.tensor([[597.6885, 597.6885]]).to(device)  # Example values
            example["context"]["index"] = torch.tensor([context_indices]).to(device)

            print(example["context"])
            print("___________")
            print(example["context"]["image"].shape)

            # Generate the Gaussians.
            visualization_dump = {}
            gaussians = self.encoder.forward(
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

