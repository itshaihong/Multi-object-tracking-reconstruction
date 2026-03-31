import sys
import os

# --- Setup Paths & Imports ---
PATH = os.getcwd()
module_dir = os.path.abspath(f"{PATH}/../FastSAM")
if module_dir not in sys.path:
    sys.path.append(module_dir)

from fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = "../tracking_dataset/shirtv1/roe1/lightbox/images/img000002.jpg"
DEVICE = 'cpu'

# 1. Manually load the image as a PIL object (matching Inference.py logic)
img_object = Image.open(IMAGE_PATH).convert("RGB")

# 2. Pass the image OBJECT, not the PATH string
everything_results = model(
    img_object, 
    device=DEVICE, 
    retina_masks=True, 
    imgsz=1024, 
    conf=0.4, 
    iou=0.9
)

# 3. Pass the image OBJECT here as well
prompt_process = FastSAMPrompt(img_object, everything_results, device=DEVICE)

# point prompt
ann = prompt_process.point_prompt(points=[[910, 530]], pointlabel=[1])

prompt_process.plot(
    annotations=ann,
    output_path="../tracking_dataset/shirtv1/roe1/lightbox/masks/img000002.jpg"
)

