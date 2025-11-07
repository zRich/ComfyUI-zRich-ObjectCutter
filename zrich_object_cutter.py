import json
import numpy as np
from PIL import Image
import torch

class ZRichObjectCutter:
    """
    🧩 ZRich Object Cutter
    将图片中的物体（通过 bbox 定位）裁剪成与原图同尺寸的透明图像。
    可直接输出给 Preview Image / Save Image 节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 与 Florence2Run 的 data 类型一致，支持直接传入其 JSON 输出
                "data": ("JSON",),
            },
            "optional": {
                "padding": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.2,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "cut_objects"
    CATEGORY = "zRich/Segmentation"

    # 裁剪物体并返回透明图像
    def cut_objects(self, image, data, padding=0.0):
        # Convert image tensor → PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(image_np)
        width, height = img_pil.size

        images = []

        # 兼容 Florence2Run 的 JSON 输出与 Florence2toCoordinates 的输入格式
        # data 可能是：
        # - JSON 字符串（包含 list 或 dict）
        # - Python 列表（[[x1,y1,x2,y2], ...] 或 [ {"bboxes": [...]}, ... ]）
        # - Python 字典（{"bboxes": [...] }）
        parsed = data
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed.replace("'", '"'))
            except Exception:
                # 如果不是有效 JSON，则当作空处理
                parsed = []

        # 提取 bboxes 列表
        bboxes = []
        if isinstance(parsed, dict) and "bboxes" in parsed:
            bboxes = parsed["bboxes"]
        elif isinstance(parsed, list):
            # 如果是列表且第一个元素是 dict，则取其中的 bboxes（兼容 batch 情况，默认取第一个）
            if len(parsed) > 0 and isinstance(parsed[0], dict) and "bboxes" in parsed[0]:
                bboxes = parsed[0]["bboxes"]
            else:
                # 否则假设就是 [[x1,y1,x2,y2], ...]
                bboxes = parsed
        else:
            bboxes = []

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [float(v) for v in box]

            # Add padding
            pad_w = (x2 - x1) * padding
            pad_h = (y2 - y1) * padding
            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(width, int(x2 + pad_w))
            y2 = min(height, int(y2 + pad_h))

            # Transparent canvas same size as original
            transparent = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            crop = img_pil.crop((x1, y1, x2, y2)).convert("RGBA")
            transparent.paste(crop, (x1, y1))

            # Convert back to tensor
            np_img = np.array(transparent).astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(np_img)[None,]
            images.append(tensor_img)

        if len(images) == 0:
            blank = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            np_blank = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(np_blank)[None,],)

        result = torch.cat(images, dim=0)
        return (result,)
