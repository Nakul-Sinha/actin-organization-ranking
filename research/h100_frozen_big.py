"""Stronger frozen features: DINOv2-large / DINOv2-giant / ConvNeXt-XXL.

The morphology signal transfers (small/base DINOv2 + CNN -> 65.57). Bigger DINOv2
backbones capture much richer structure -> stronger per-tile organization score.
Saves per-model logits to combine with the CNN ensemble into a better grand.
"""
import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "research")
import h100_frozen as HF

HF.OUT = Path("cands4"); HF.OUT.mkdir(exist_ok=True)
HF.MODELS = [
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
]

if __name__ == "__main__":
    HF.main()
