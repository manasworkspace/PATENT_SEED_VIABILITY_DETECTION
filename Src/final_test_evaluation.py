import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

FINAL_THRESHOLD = 0.10
NUM_CLASSES = 2

MODEL_PATH = "checkpoints/fasterrcnn_epoch5.pth"

TEST_IMAGES = "data/test/images"

TEST_ANN = "data/test/binary_annotations.json"


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded successfully")


transform = T.Compose([T.ToTensor()])

test_dataset = CocoDetection(
    root=TEST_IMAGES,
    annFile=TEST_ANN,
    transform=transform
)

def collate_fn(batch):
    return tuple(zip(*batch))

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

print("Test dataset loaded")



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

TP = 0
FP = 0
FN = 0

coco_results = []

with torch.no_grad():
    for idx, (images, targets) in enumerate(tqdm(test_loader)):

        images_gpu = [img.to(DEVICE) for img in images]
        outputs = model(images_gpu)

        preds = outputs[0]
        scores = preds["scores"].cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()

        keep = scores >= FINAL_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]

        image_id = test_dataset.ids[idx]

        for box, score in zip(boxes, scores):
            coco_results.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1])
                ],
                "score": float(score)
            })

        gt_boxes = []
        for ann in targets[0]:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x+w, y+h])

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

print("\n==============================")
print("FINAL TEST METRICS")
print("==============================")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")



coco_gt = COCO(TEST_ANN)
coco_dt = coco_gt.loadRes(coco_results)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()