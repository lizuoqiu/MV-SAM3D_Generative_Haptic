import sys
from pathlib import Path

# import inference code
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR / "notebook"))
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = ROOT_DIR / "checkpoints" / tag / "pipeline.yaml"
inference = Inference(str(config_path), compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image(str(ROOT_DIR / "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"))
mask = load_single_mask(str(ROOT_DIR / "notebook/images/shutterstock_stylish_kidsroom_1640806567"), index=11)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(str(ROOT_DIR / "splat.ply"))
print("Your reconstruction has been saved to splat.ply")
