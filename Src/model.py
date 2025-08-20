import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes=2, pretrained=True, device=None):
    """
    Returns a Faster R-CNN model with custom number of classes.
    
    Args:
        num_classes (int): Number of output classes (including background)
        pretrained (bool): Use COCO pretrained weights
        device (torch.device): GPU/CPU device
    """
   
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained
    )

   
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device