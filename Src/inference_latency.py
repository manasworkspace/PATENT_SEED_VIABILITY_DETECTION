import torch
import time
import os
from PIL import Image
from torchvision import transforms

from src.model import get_faster_rcnn_model



MODEL_PATH = "checkpoints/fasterrcnn_epoch5.pth"

TEST_IMAGE_DIR = "data/test/images"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model, _  = get_faster_rcnn_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()



transform = transforms.Compose([
    transforms.ToTensor()
])



image_files = os.listdir(TEST_IMAGE_DIR)
image_files = [f for f in image_files if f.endswith(".jpg") or f.endswith(".png")]

print("Total test images:", len(image_files))



total_time = 0

with torch.no_grad():

    for img_name in image_files:

        img_path = os.path.join(TEST_IMAGE_DIR, img_name)

        image = Image.open(img_path).convert("RGB")
        image = transform(image).to(device)

        start = time.time()

        outputs = model([image])

        end = time.time()

        latency = end - start
        total_time += latency



avg_latency = total_time / len(image_files)
fps = 1 / avg_latency

print("\n----- Inference Latency Results -----")
print(f"Average Latency: {avg_latency:.4f} seconds per image")
print(f"FPS: {fps:.2f}")