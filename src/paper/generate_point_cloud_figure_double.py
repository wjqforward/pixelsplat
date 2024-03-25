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
from jaxtyping import Float, Int64
import time

SCENES = (
    # scene, [context 1, context 2], far plane
    # ("test", [785, 795], 15, [0]),
    # ("1825_1865", [1825, 1865], 15, [0, 30, 60, 90, 120, 150]),
    # ("124_128", [124, 128], 15, [0, 30, 60, 90, 120, 150]),
    ("512_522", [0, 9], 10, [0]),
    ("512_522", [10, 19], 10, [0]),
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

    
    # camera_params = data['camera']
    # fx = camera_params['fx']
    # fy = camera_params['fy']
    # cx = camera_params['cx']
    # cy = camera_params['cy']

    fx = 600.0 / 1200
    fy = 600.0 / 680
    cx = 0.5
    cy = 0.5


    intrinsics = torch.tensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=torch.float32)
    # normalized_cx = cx /fx
    # normalized_cy = cy /fy
    # normalized_fx = fx /fx
    # normalized_fy = fy /fy

    # intrinsics = torch.tensor([[normalized_fx, 0, normalized_cx],
    #                         [0, normalized_fy, normalized_cy],
    #                         [0, 0, 1]], dtype=torch.float32)

    # intrinsics = torch.tensor([[1.0118, 0.0000, 0.5000],
    #       [0.0000, 1.0120, 0.5000],
    #       [0.0000, 0.0000, 1.0000]], dtype=torch.float32)
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

    print("ex ", extrinsics_cuda)
    return extrinsics_cuda



def load_images(indices):
    
    preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    images = []
    image_paths = [f"{scene_path}frame{index:06}.jpg" for index in indices]

    for path in image_paths:
        image = Image.open(path)  
        image_tensor = preprocess(image)
        # save_image(image_tensor, 'new_image_path.jpg')
        image_tensor = image_tensor.unsqueeze(0) 
        images.append(image_tensor)

    batch_tensor = torch.cat(images, dim=0)
    batch_tensor_5d = batch_tensor.unsqueeze(0)
    
    # batch_tensor_5d = batch_tensor_5d.to(device)
    
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

    gs1 = None

    count = 0
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
        # print(example["context"]["extrinsics"])
        # time.sleep(10)
        example["context"]["intrinsics"] = intrinsics.unsqueeze(0)
        example["context"]["image"] = load_images(context_indices).to(device)
        # Assuming near and far values are predefined or calculated elsewhere
        example["context"]["near"] = torch.tensor([[0.05, 0.05]]).to(device)  # Example values
        example["context"]["far"] = torch.tensor([[10.0, 10.0]]).to(device)  # Example values
        example["context"]["index"] = torch.tensor([context_indices]).to(device)

        # print(example["context"])
        # print("___________")
        # print(example["context"]["image"].shape)

        # Generate the Gaussians.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"], True, visualization_dump=visualization_dump
        )
        time1 = time.time()
        print(time1-start_time)

        op_mask = gaussians.opacities < 0.15
        gaussians.means = gaussians.means[~op_mask].unsqueeze(0)
        gaussians.covariances = gaussians.covariances[~op_mask].unsqueeze(0)
        gaussians.harmonics = gaussians.harmonics[~op_mask].unsqueeze(0)
        gaussians.opacities = gaussians.opacities[~op_mask].unsqueeze(0)
        # print(gaussians.covariances.shape)
        # print(gaussians.covariances)
        # print(gaussians.harmonics.shape)
        # print(gaussians.harmonics)
        # print("TESTTESTTEST")
        # 计算每个协方差矩阵对角线元素（方差）的平均值
        variance_means = gaussians.covariances.diagonal(dim1=-2, dim2=-1).mean(dim=-1)

        # 创建一个全0的张量，形状与原始协方差矩阵相同
        isotropic_covariances = torch.zeros_like(gaussians.covariances)

        # 填充对角线元素为方差的平均值，创建各向同性协方差矩阵
        # 这里使用了广播机制：variance_means[..., None, None] 将平均方差扩展到合适的维度
        isotropic_covariances[..., 0, 0] = variance_means
        isotropic_covariances[..., 1, 1] = variance_means
        isotropic_covariances[..., 2, 2] = variance_means

        # 更新原始的协方差矩阵为各向同性协方差矩阵
        gaussians.covariances = isotropic_covariances

        zeroed_sh_coefficients = torch.zeros_like(gaussians.harmonics)

        # 仅将0阶系数的值从原始张量复制到新的全0张量中
        # 0阶系数对应最后一个维度的第一个元素
        zeroed_sh_coefficients[..., 0] = gaussians.harmonics[..., 0]
        gaussians.harmonics = zeroed_sh_coefficients
        # print("99999999999999999")
        # print(gaussians.harmonics)
        # print(")))))))))))))))))))))))))))))))")

        time2 = time.time()
        print(time2-start_time)

        if count == 0:
            gs1 = gaussians
        
        count += 1


    gaussians.means = torch.cat((gaussians.means, gs1.means), dim=1)
    gaussians.covariances = torch.cat((gaussians.covariances, gs1.covariances), dim=1)
    gaussians.harmonics = torch.cat((gaussians.harmonics, gs1.harmonics), dim=1)
    gaussians.opacities = torch.cat((gaussians.opacities, gs1.opacities), dim=1)
    print(gaussians.covariances.shape)
    # print(gaussians.covariances)
    # print(gaussians.harmonics.shape)
    # print(gaussians.harmonics)
    # print("TESTTESTTEST")
    # Render depth.
    *_, h, w = example["context"]["image"].shape
    rendered = decoder.forward(
        gaussians,
        example["context"]["extrinsics"],
        example["context"]["intrinsics"],
        example["context"]["near"],
        example["context"]["far"],
        (h, w),
        "depth",
    )

    time3 = time.time()
    print(time3-time2)
    target_gt = example["context"]["image"]

    # Compute metrics.
    psnr_probabilistic = compute_psnr(
        rearrange(target_gt, "b v c h w -> (b v) c h w"),
        rearrange(rendered.color, "b v c h w -> (b v) c h w"),
    )
    print("train/psnr_probabilistic", psnr_probabilistic.mean())

    print(rendered.color.shape)

    for i in range(rendered.color.size(1)):
        save_image(rendered.color[0, i], f'image_{i}.png')

    for i in range(target_gt.size(1)):
        save_image(target_gt[0, i], f'gt_{i}.png')



if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
