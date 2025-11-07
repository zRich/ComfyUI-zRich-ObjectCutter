from .zrich_object_cutter import ZRichObjectCutter
from .sam2_auto_layer_node import SAM2AutoLayerNode

NODE_CLASS_MAPPINGS = {
    "zRichObjectCutter": ZRichObjectCutter,
    "SAM2AutoLayer": SAM2AutoLayerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zRichObjectCutter": "🧩 zRich Object Cutter (MASK to Transparent)",
    "SAM2AutoLayer": "SAM2 Auto Layer Extractor",
}
