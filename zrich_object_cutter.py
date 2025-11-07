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

        # 如果提供了 bboxes，则依据每个框把原始掩码“拆分”为多个全尺寸掩码：仅保留框内像素
        # 构建并集掩码，作为拆分的基底
        union_alpha_for_split = np.zeros((H, W), dtype=np.float32)
        if masks_np.shape[0] > 0:
            for i in range(masks_np.shape[0]):
                union_alpha_for_split = np.maximum(union_alpha_for_split, (masks_np[i] > 0.5).astype(np.float32))

        # 解析 bboxes：支持 [ [x1,y1,x2,y2], ... ] 或按批次嵌套结构
        boxes_for_split = []
        try:
            if isinstance(bboxes, torch.Tensor):
                bb = bboxes.detach().cpu().numpy()
            else:
                bb = np.array(bboxes, dtype=np.int64)
            if bb.ndim == 1 and bb.shape[0] == 4:
                boxes_for_split = [bb.tolist()]
            elif bb.ndim >= 2:
                reshaped = bb.reshape(-1, bb.shape[-1])
                if reshaped.shape[-1] == 4:
                    boxes_for_split = reshaped.tolist()
        except Exception:
            boxes_for_split = []

        # 若有框，则按框生成全尺寸掩码（掩码与原图尺寸一致，仅保留框内像素），并排除交叉区域
        # 规则：按框顺序赋予像素归属，同一像素只归属于第一个覆盖它的框
        if len(boxes_for_split) > 0:
            def clamp_int(v, lo, hi):
                return int(max(lo, min(hi, v)))
            split_masks = []
            assigned = np.zeros((H, W), dtype=np.uint8)  # 已分配像素标记（0/1）
            union_alpha_for_split_bin = (union_alpha_for_split > 0.5).astype(np.uint8)
            for bx in boxes_for_split:
                x1, y1, x2, y2 = bx
                x1 = clamp_int(x1, 0, W)
                x2 = clamp_int(x2, 0, W)
                y1 = clamp_int(y1, 0, H)
                y2 = clamp_int(y2, 0, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                # 候选像素：并集掩码中的像素
                candidate = union_alpha_for_split_bin[y1:y2, x1:x2]
                # 排除已分配像素，确保不产生交叉
                exclusive = candidate * (1 - assigned[y1:y2, x1:x2])
                # 将本框的独占像素写入全尺寸掩码
                full_mask = np.zeros((H, W), dtype=np.float32)
                full_mask[y1:y2, x1:x2] = exclusive.astype(np.float32)
                split_masks.append(full_mask)
                # 标记这些像素为已分配
                assigned[y1:y2, x1:x2] = np.maximum(assigned[y1:y2, x1:x2], exclusive)
            if len(split_masks) > 0:
                masks_np = np.stack(split_masks, axis=0)
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

        # 第二路输出：如果提供了 bboxes，则按框输出只包含该框独占像素的全尺寸透明图
        crop_outputs = []
        def clamp_int(v, lo, hi):
            return int(max(lo, min(hi, v)))

        # 直接复用 boxes_for_split，确保与 masks_np 的顺序一致
        boxes = boxes_for_split

        if boxes:
            for i, bx in enumerate(boxes):
                x1, y1, x2, y2 = bx
                # 边界裁剪并保证有效区域
                x1 = clamp_int(x1, 0, W)
                x2 = clamp_int(x2, 0, W)
                y1 = clamp_int(y1, 0, H)
                y2 = clamp_int(y2, 0, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                # 使用每个框对应的独占掩码生成 RGBA，并仅在框区域拷贝
                alpha_i = (masks_np[i] > 0.5).astype(np.float32)
                rgba_i = np.concatenate([src * alpha_i[..., None], alpha_i[..., None]], axis=-1)
                canvas = np.zeros((H, W, 4), dtype=np.float32)
                canvas[y1:y2, x1:x2, :] = rgba_i[y1:y2, x1:x2, :].astype(np.float32)
                crop_outputs.append(torch.from_numpy(canvas).unsqueeze(0))

        if crop_outputs:
            crops_batch = torch.cat(crop_outputs, dim=0)
        else:
            # 没有提供 bboxes 或解析失败时，复用整幅抠图（第一路）
            crops_batch = per_mask_rgba

        return (per_mask_rgba, crops_batch)
