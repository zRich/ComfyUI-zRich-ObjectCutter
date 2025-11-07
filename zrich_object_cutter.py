import json
import numpy as np
from PIL import Image
import torch

class ZRichObjectCutter:
    """
    🧩 ZRich Object Cutter
    将图片中的物体（通过 MASK）抠图为与原图同尺寸的透明图像。
    现在支持直接接收 SAM2 的 MASK 输出，逐个区域生成透明背景 RGBA 图片。
    可选地接收 bboxes，用掩码合成后的图按框逐个裁剪输出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 直接接收 Sam2Segmentation 的输出 MASK
                "mask": ("MASK",),
                # 改为必填，与 Florence2toCoordinates 的 BBOX 输出一致
                "bboxes": ("BBOX",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "crops")
    FUNCTION = "cut_objects"
    CATEGORY = "zRich/Segmentation"

    # 基于 MASK 抠图并返回透明 RGBA 图像（尺寸与原图一致）
    def cut_objects(self, image, mask, bboxes):
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
            blank_t = torch.from_numpy(blank).unsqueeze(0)
            return (blank_t, blank_t)

        # 第一路输出：每个 mask 的整幅透明抠图
        per_mask_rgba = torch.cat(outputs, dim=0)  # (N, H, W, 4)

        # 第二路输出：如果提供了 bboxes，则按框从合成图裁剪；否则复用第一路
        # 合成图：掩码并集后的整体抠图（避免多对象被覆盖为黑）
        union_alpha = np.zeros((H, W), dtype=np.float32)
        for i in range(masks_np.shape[0]):
            union_alpha = np.maximum(union_alpha, (masks_np[i] > 0.5).astype(np.float32))
        union_rgb = src * union_alpha[..., None]
        union_rgba = np.concatenate([union_rgb, union_alpha[..., None]], axis=-1)  # (H,W,4)

        crop_outputs = []
        def clamp_int(v, lo, hi):
            return int(max(lo, min(hi, v)))

        # 解析 bboxes：支持 [ [x1,y1,x2,y2], ... ] 或按批次嵌套结构
        boxes = []
        try:
            # torch/numpy/list 统一为 Python 列表
            if isinstance(bboxes, torch.Tensor):
                bb = bboxes.detach().cpu().numpy()
            else:
                bb = np.array(bboxes, dtype=np.int64)
            # 尝试展平到 (M,4)
            if bb.ndim == 1 and bb.shape[0] == 4:
                boxes = [bb.tolist()]
            elif bb.ndim >= 2:
                # 如果是按批次嵌套，则取第一维的所有框或直接重塑到 (-1,4)
                reshaped = bb.reshape(-1, bb.shape[-1])
                if reshaped.shape[-1] == 4:
                    boxes = reshaped.tolist()
        except Exception:
            boxes = []

        if boxes:
            for bx in boxes:
                x1, y1, x2, y2 = bx
                # 边界裁剪并保证有效区域
                x1 = clamp_int(x1, 0, W)
                x2 = clamp_int(x2, 0, W)
                y1 = clamp_int(y1, 0, H)
                y2 = clamp_int(y2, 0, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                # 每个 bbox 输出与原图同尺寸的透明图，只在框区域拷贝像素
                canvas = np.zeros((H, W, 4), dtype=np.float32)
                canvas[y1:y2, x1:x2, :] = union_rgba[y1:y2, x1:x2, :].astype(np.float32)
                crop_outputs.append(torch.from_numpy(canvas).unsqueeze(0))

        if crop_outputs:
            crops_batch = torch.cat(crop_outputs, dim=0)
        else:
            # 没有提供 bboxes 或解析失败时，复用整幅抠图（第一路）
            crops_batch = per_mask_rgba

        return (per_mask_rgba, crops_batch)
