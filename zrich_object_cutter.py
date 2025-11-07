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
        # 输入 image: (B, H, W, C) 浮点 0..1；mask: (N, H, W) 或 (H, W)
        img_np = image.cpu().numpy().astype(np.float32)  # 保持 0..1
        B, H, W, C = img_np.shape

        mask_np = mask.cpu().numpy()

        # 统一为列表形式的掩码
        masks = []
        if mask_np.ndim == 2:
            masks = [mask_np]
        elif mask_np.ndim == 3:
            masks = [mask_np[i] for i in range(mask_np.shape[0])]
        else:
            masks = []

        outputs = []
        for i, m in enumerate(masks):
            # 选择对应图像；若 mask 数量与图像批次一致则一一对应，否则默认使用第 0 张原图
            if len(masks) == B:
                src = img_np[i]
            else:
                src = img_np[0]

            # 二值化掩码并做透明背景
            alpha = (m > 0.5).astype(np.float32)
            rgb = src * alpha[..., None]
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)

            tensor_img = torch.from_numpy(rgba)[None,]
            outputs.append(tensor_img)

        if len(outputs) == 0:
            # 如果没有掩码，输出一张透明图（与第一张原图同尺寸）
            blank = np.zeros((H, W, 4), dtype=np.float32)
            return (torch.from_numpy(blank)[None,],)

        result = torch.cat(outputs, dim=0)
        return (result,)
