import os
import traceback
import logging
import numpy as np
import torch

from typing import List, Dict, Tuple, Optional

import folder_paths

logger = logging.getLogger(__name__)

# 延迟导入 SAM2 依赖（来自 comfyui-segment-anything-2 插件的 sam2 包）
SAM2_IMPORTED = False
SAM2_IMPORT_ERROR = None
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_IMPORTED = True
except Exception as e:
    SAM2_IMPORT_ERROR = str(e)
    logger.error(f"SAM2 模块导入失败: {e}")


# 配置文件映射
SAM2_CONFIG_MAPPING = {
    "hiera_tiny": "sam2_hiera_t.yaml",
    "hiera_small": "sam2_hiera_s.yaml",
    "hiera_base_plus": "sam2_hiera_b+.yaml",
    "hiera_large": "sam2_hiera_l.yaml",
}

DEFAULT_CONFIG = "sam2_hiera_b+.yaml"


def _segment_anything2_configs_dir() -> str:
    """定位 comfyui-segment-anything-2 插件中的配置目录（sam2_configs）。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(script_dir)
    cand = os.path.join(custom_nodes_dir, "comfyui-segment-anything-2", "sam2_configs")
    return cand


def get_config_path(model_name: str) -> Optional[str]:
    """根据模型名称返回配置文件路径，优先使用 comfyui-segment-anything-2 的 sam2_configs。"""
    # 根据模型名称推断配置文件
    config_filename = DEFAULT_CONFIG
    lower_name = model_name.lower()
    for key, cfg in SAM2_CONFIG_MAPPING.items():
        if key in lower_name:
            config_filename = cfg
            break

    # 优先使用 comfyui-segment-anything-2 的配置目录
    cfg_dir = _segment_anything2_configs_dir()
    candidate = os.path.join(cfg_dir, config_filename)
    if os.path.exists(candidate):
        return candidate

    # 兜底：尝试本子模块下的 configs 目录（如存在）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_cfg_dir = os.path.join(script_dir, "configs")
    candidate2 = os.path.join(local_cfg_dir, config_filename)
    if os.path.exists(candidate2):
        return candidate2

    logger.warning(f"未找到配置文件 {config_filename}，请确保 comfyui-segment-anything-2 已安装并包含 sam2_configs。")
    return None


def get_sam2_models() -> List[str]:
    """获取可用的 SAM2 模型列表（基于 ComfyUI 的 folder_paths）。"""
    models: List[str] = []
    try:
        sam2_models = folder_paths.get_filename_list("sam2")
        if sam2_models:
            models.extend(sam2_models)
    except Exception:
        pass
    try:
        sams_models = folder_paths.get_filename_list("sams")
        if sams_models:
            models.extend(sams_models)
    except Exception:
        pass
    models = sorted(list(set(models)))
    if not models:
        models = ["sam2_hiera_base_plus.safetensors"]
    return models


def get_model_path(model_name: str) -> Optional[str]:
    """返回模型文件的完整路径。"""
    try:
        p = folder_paths.get_full_path("sam2", model_name)
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    try:
        p = folder_paths.get_full_path("sams", model_name)
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return None


class SAM2AutoLayerNode:
    """SAM2 自动图层提取节点（位于 ComfyUI-zRich-ObjectCutter 子模块中）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (
                    get_sam2_models(),
                    {
                        "default": get_sam2_models()[0] if get_sam2_models() else "sam2_hiera_base_plus.safetensors"
                    },
                ),
                "min_area": ("INT", {"default": 5000, "min": 1, "max": 99999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cutout_layers",)
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    # 类级别的模型缓存
    _mask_generator = None
    _device = None
    _cached_model_name = None
    _cached_config_path = None

    @classmethod
    def clear_cache(cls):
        cls._mask_generator = None
        cls._device = None
        cls._cached_model_name = None
        cls._cached_config_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_inputs(self, image: torch.Tensor, model_name: str, min_area: int) -> Tuple[bool, str]:
        if image is None or len(image.shape) < 3:
            return False, "输入图像无效"
        if not model_name or not isinstance(model_name, str):
            return False, "模型名称无效"
        if min_area < 1:
            return False, "最小区域面积必须大于0"
        return True, ""

    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        if len(image_np.shape) == 3:
            if image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
            elif image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)
        elif len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
        return image_np

    def _should_reload(self, model_name: str, device: torch.device) -> bool:
        return (
            self._mask_generator is None
            or self._device != device
            or self._cached_model_name != model_name
        )

    def _load_model(self, model_name: str, device: torch.device) -> bool:
        try:
            model_path = get_model_path(model_name)
            if model_path is None:
                logger.error(f"无法找到模型文件: {model_name}")
                return False

            config_path = get_config_path(model_name)
            if config_path is None or not os.path.exists(config_path):
                logger.error(f"无法找到配置文件: {config_path}")
                return False

            logger.info(f"加载 SAM2 模型: {model_name}")
            logger.info(f"模型路径: {model_path}")
            logger.info(f"配置路径: {config_path}")

            # 直接使用 build_sam2（来自 comfyui-segment-anything-2 的 sam2 包）
            sam2_model = build_sam2(os.path.basename(config_path), model_path, device=device)
            self._mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            self._device = device
            self._cached_model_name = model_name
            self._cached_config_path = config_path
            return True
        except Exception as e:
            logger.error(f"SAM2 模型加载失败: {e}")
            traceback.print_exc()
            return False

    def _generate_cutout_layers(self, image_np: np.ndarray, masks: List[Dict]) -> torch.Tensor:
        if len(masks) == 0:
            rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = image_np
            rgba[..., 3] = 255
            return torch.tensor(rgba / 255.0, dtype=torch.float32).unsqueeze(0)

        layers = []
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                continue
            rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = image_np
            rgba[..., 3] = (seg * 255).astype(np.uint8)
            layers.append(torch.tensor(rgba / 255.0, dtype=torch.float32))
        if len(layers) == 0:
            rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            return torch.tensor(rgba / 255.0, dtype=torch.float32).unsqueeze(0)
        return torch.stack(layers, dim=0)

    def segment(self, image: torch.Tensor, model_name: str, min_area: int) -> Tuple[torch.Tensor]:
        logger.info(f"SAM2AutoLayerNode.segment: model_name={model_name}, min_area={min_area}")
        if not SAM2_IMPORTED:
            err = f"SAM2 模块未导入，无法执行分层。错误: {SAM2_IMPORT_ERROR}"
            logger.error(err)
            return (torch.zeros(1, 64, 64, 4, dtype=torch.float32),)

        valid, msg = self._validate_inputs(image, model_name, min_area)
        if not valid:
            logger.error(f"输入验证失败: {msg}")
            return (torch.zeros(1, 64, 64, 4, dtype=torch.float32),)

        try:
            image_np = self._prepare_image(image)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._should_reload(model_name, device):
                if not self._load_model(model_name, device):
                    return (torch.zeros(1, 64, 64, 4, dtype=torch.float32),)

            logger.info("开始生成分割掩码...")
            masks = self._mask_generator.generate(image_np)
            filtered = [m for m in masks if m.get("area", 0) >= min_area]
            logger.info(f"生成了 {len(masks)} 个掩码，过滤后剩余 {len(filtered)} 个")
            layers = self._generate_cutout_layers(image_np, filtered)
            return (layers,)
        except Exception as e:
            logger.error(f"SAM2 分割过程中发生错误: {e}")
            traceback.print_exc()
            return (torch.zeros(1, 64, 64, 4, dtype=torch.float32),)