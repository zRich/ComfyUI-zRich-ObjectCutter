import json
import numpy as np
from PIL import Image
import torch

class ZRichObjectCutter:
    """
    🧩 ZRich Object Cutter
    将图片中的物体（通过 MASK 或 bbox）抠图为与原图同尺寸的透明图像。
    现在支持直接接收 SAM2 的 MASK 输出，逐个区域生成透明背景 RGBA 图片。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 直接接收 Sam2Segmentation 的输出 MASK
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "cut_objects"
    CATEGORY = "zRich/Segmentation"

    # 基于 MASK 抠图并返回透明 RGBA 图像（尺寸与原图一致）
    def cut_objects(self, image, mask):
        # 输入 image: (B, H, W, C) 浮点 0..1；mask: (..., H, W)
        img_np = image.detach().cpu().numpy().astype(np.float32)  # 保持 0..1
        B, H, W, C = img_np.shape

        mask_np = mask.detach().cpu().numpy().astype(np.float32)

        # 统一为 (N, H, W)：将除最后两个维度外的所有前导维度展平为 N
        if mask_np.ndim == 2:
            masks_np = mask_np.reshape(1, H, W)
        elif mask_np.ndim >= 3:
            h, w = mask_np.shape[-2], mask_np.shape[-1]
            n = int(np.prod(mask_np.shape[:-2]))
            masks_np = mask_np.reshape(n, h, w)
        else:
            masks_np = np.zeros((0, H, W), dtype=np.float32)

        outputs = []
        # 始终使用第一张原图进行抠图；对每个 mask 输出一张透明图
        src = img_np[0]  # (H, W, C)
        for i in range(masks_np.shape[0]):
            m = masks_np[i]

            # 二值化掩码并做透明背景（按 0/1 阈值）
            alpha = (m > 0.5).astype(np.float32)
            rgb = src * alpha[..., None]
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)

            outputs.append(torch.from_numpy(rgba).unsqueeze(0))  # (1, H, W, 4)

        if len(outputs) == 0:
            # 如果没有掩码，输出一张透明图（与第一张原图同尺寸）
            blank = np.zeros((H, W, 4), dtype=np.float32)
            return (torch.from_numpy(blank).unsqueeze(0),)

        result = torch.cat(outputs, dim=0)  # (N, H, W, 4)
        return (result,)
