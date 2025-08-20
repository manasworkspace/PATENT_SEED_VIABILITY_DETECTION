import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.dataset import BinaryCocoDataset
from src.model import get_faster_rcnn_model
from torchvision import transforms as T
import os

transform = T.Compose([
    T.ToTensor(),
])


DATA_ROOT = "data"   
NUM_CLASSES = 2
BATCH_SIZE = 2
NUM_EPOCHS = 5
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


train_dataset = BinaryCocoDataset(
    images_dir=f"{DATA_ROOT}/train/images",
    annotation_file=f"{DATA_ROOT}/train/binary_annotations.json",
    transforms=transform
)

valid_dataset = BinaryCocoDataset(
    images_dir=f"{DATA_ROOT}/valid/images",
    annotation_file=f"{DATA_ROOT}/valid/binary_annotations.json",
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, device = get_faster_rcnn_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
    device=device
)


params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)


for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0.0

    for images, targets in train_loader:

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    lr_scheduler.step()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"fasterrcnn_epoch{epoch+1}.pth"
    )

    torch.save(model.state_dict(), checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")

print("Training completed!")