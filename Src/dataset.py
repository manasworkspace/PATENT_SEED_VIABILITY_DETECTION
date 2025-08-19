import json
from collections import defaultdict

def inspect_coco_dataset(annotation_path):
    """
    Inspects a COCO-format dataset and prints dataset statistics.
    """

    print(f"\nInspecting: {annotation_path}")

    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    print(f"\nTotal Images: {len(images)}")
    print(f"Total Annotations: {len(annotations)}")

   
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    class_counts = defaultdict(int)

    for ann in annotations:
        cat_id = ann["category_id"]
        class_name = category_id_to_name.get(cat_id, "Unknown")
        class_counts[class_name] += 1

    print("\nClass Distribution:")
    print("-" * 30)
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")

    print("-" * 30)

def convert_to_binary_coco(input_path, output_path):
    """
    Converts a multi-class COCO dataset into binary classification:
    0 -> Non-Viable
    1 -> Viable
    """

    import json

    print(f"\nConverting: {input_path}")

    with open(input_path, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])

   
    binary_mapping = {}

    for cat in categories:
        name = cat["name"]

        if name in ["Seeds", "Day0", "Non-Germinated"]:
            binary_mapping[cat["id"]] = 0  

        elif name in ["Germinated", "Sprout"]:
            binary_mapping[cat["id"]] = 1  

        else:
            raise ValueError(f"Unknown category: {name}")


    for ann in coco_data["annotations"]:
        old_id = ann["category_id"]
        ann["category_id"] = binary_mapping[old_id]

    
    coco_data["categories"] = [
        {"id": 0, "name": "Non-Viable"},
        {"id": 1, "name": "Viable"}
    ]

    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"Saved binary dataset to: {output_path}")


import torch
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image

class BinaryCocoDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]
        path = f"{self.images_dir}/{img_info['file_name']}"
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        if self.transforms:
            img = self.transforms(img)

        return img, target