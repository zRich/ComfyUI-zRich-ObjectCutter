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
                "bboxes": ("LIST",),  # [[x1,y1,x2,y2], ...]
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
    RETURN_NAMES = ("objects",)
    FUNCTION = "cut_objects"
    CATEGORY = "ZRich/Segmentation"

    def cut_objects(self, image, bboxes, padding=0.0):
        # Convert image tensor → PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(image_np)
        width, height = img_pil.size

        objects = []

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
            objects.append(tensor_img)

        if len(objects) == 0:
            blank = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            np_blank = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(np_blank)[None,],)

        result = torch.cat(objects, dim=0)
        return (result,)
