import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from tqdm import tqdm



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.cuda.empty_cache()



DATA_ROOT = "data"
CHECKPOINT_PATH = "checkpoints/fasterrcnn_epoch5.pth"



NUM_CLASSES = 2  # background + seed class

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")


transform = T.Compose([
    T.ToTensor()
])

val_dataset = CocoDetection(
    root=f"{DATA_ROOT}/valid/images",
    annFile=f"{DATA_ROOT}/valid/binary_annotations.json",
    transform=transform
)

def collate_fn(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

print("Validation dataset loaded")



def compute_iou(box1, box2):

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    unionArea = box1Area + box2Area - interArea

    return interArea / unionArea if unionArea > 0 else 0



thresholds = np.arange(0.1, 0.9, 0.1)

best_f1 = 0
best_threshold = 0

for threshold in thresholds:

    TP = 0
    FP = 0
    FN = 0

    print(f"\nEvaluating Threshold: {threshold:.2f}")

    with torch.no_grad():

        for images, targets in tqdm(val_loader):

            images = [img.to(DEVICE) for img in images]

            outputs = model(images)

            preds = outputs[0]

            scores = preds["scores"].cpu().numpy()
            boxes = preds["boxes"].cpu().numpy()

            keep = scores >= threshold
            boxes = boxes[keep]

            gt_boxes = []

            for ann in targets[0]:
                x, y, w, h = ann["bbox"]
                gt_boxes.append([x, y, x + w, y + h])

            matched = []

            for pred_box in boxes:

                found_match = False

                for i, gt_box in enumerate(gt_boxes):

                    if i in matched:
                        continue

                    iou = compute_iou(pred_box, gt_box)

                    if iou >= 0.5:
                        TP += 1
                        matched.append(i)
                        found_match = True
                        break

                if not found_match:
                    FP += 1

            FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold



print("\n==============================")
print("BEST THRESHOLD FOUND")
print("==============================")
print(f"Threshold: {best_threshold:.2f}")
print(f"Best F1:   {best_f1:.4f}")