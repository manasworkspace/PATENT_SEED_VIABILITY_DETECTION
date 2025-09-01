import torch
from torch.utils.data import DataLoader
from src.dataset import BinaryCocoDataset
from src.model import get_faster_rcnn_model
from torchvision.ops import box_iou
from collections import defaultdict
from torchvision import transforms as T
import os


transform = T.Compose([
    T.ToTensor()
])


DATA_ROOT = "data"
CHECKPOINT_PATH = "checkpoints/fasterrcnn_epoch5.pth"


NUM_CLASSES = 2  # Non-Viable, Viable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESHOLD = 0.5


model, _ = get_faster_rcnn_model(
    num_classes=NUM_CLASSES,
    pretrained=False,
    device=DEVICE
)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()


test_dataset = BinaryCocoDataset(
    images_dir=f"{DATA_ROOT}/test/images",
    annotation_file=f"{DATA_ROOT}/test/binary_annotations.json",
    transforms=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)


TP = defaultdict(int)
FP = defaultdict(int)
FN = defaultdict(int)

for images, targets in test_loader:

    images = [img.to(DEVICE) for img in images]
    outputs = model(images)

    for output, target in zip(outputs, targets):

        pred_boxes = output["boxes"].detach().cpu()
        pred_labels = output["labels"].detach().cpu()

        true_boxes = target["boxes"].cpu()
        true_labels = target["labels"].cpu()

        for cls in range(1, NUM_CLASSES):

            cls_true = true_boxes[true_labels == cls]
            cls_pred = pred_boxes[pred_labels == cls]

            if len(cls_true) == 0 and len(cls_pred) == 0:
                continue

            if len(cls_pred) > 0 and len(cls_true) > 0:
                ious = box_iou(cls_pred, cls_true)
            else:
                ious = torch.zeros((len(cls_pred), len(cls_true)))

            matched_true = set()

            for i in range(len(cls_pred)):

                if len(cls_true) == 0:
                    FP[cls] += 1
                    continue

                iou_vals, idxs = ious[i].max(0)

                if iou_vals >= IOU_THRESHOLD and idxs.item() not in matched_true:
                    TP[cls] += 1
                    matched_true.add(idxs.item())
                else:
                    FP[cls] += 1

            FN[cls] += len(cls_true) - len(matched_true)


for cls in range(1, NUM_CLASSES):

    precision = TP[cls] / (TP[cls] + FP[cls]) if (TP[cls] + FP[cls]) > 0 else 0.0
    recall = TP[cls] / (TP[cls] + FN[cls]) if (TP[cls] + FN[cls]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    label_name = "Viable" if cls == 1 else "Non-Viable"

    print(f"{label_name} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

print("Evaluation completed!")