

---

#  Training Pipeline

The training pipeline is designed to fine-tune an object detection model for identifying **viable** and **non-viable seeds** from annotated seed images. The system is implemented using **PyTorch** and **Torchvision**, leveraging transfer learning from a pretrained **Faster R-CNN** model.

---

##  1. Dataset Preparation

The dataset contains annotated seed images with bounding boxes identifying seed locations and their viability status.

### Dataset Split

| Split          | Number of Images |
| -------------- | ---------------- |
| Training       | 5733             |
| Validation     | 1642             |
| **Total Used** | **7375**         |

Each annotation contains:

* Bounding box coordinates
* Class label:

  * **Viable Seed**
  * **Non-Viable Seed**

The dataset is loaded using a custom PyTorch dataset loader that converts the annotations into the format required by the Faster R-CNN detection model.

---

##  2. Data Loading

The dataset is loaded using the PyTorch `Dataset` and `DataLoader` classes.

Key steps include:

* Converting images to tensors
* Formatting bounding boxes
* Handling variable numbers of objects per image
* Using a custom **collate function** to allow batching of object detection data

---

##  3. Model Initialization

The model is initialized using a pretrained **Faster R-CNN** with a **ResNet-50 FPN** backbone from **Torchvision**.

Transfer learning is applied using weights pretrained on the **COCO Dataset**.

The final classification head is replaced to support the custom classes used in this project.

### Classes Used

```
Background
Viable Seed
Non-Viable Seed
```

---

##  4. Model Training

The model is trained on the training dataset using the following procedure:

1. Load pretrained Faster R-CNN model
2. Replace the classification head with a custom predictor
3. Forward pass images through the model
4. Compute detection losses:

   * classification loss
   * bounding box regression loss
5. Perform backpropagation
6. Update model weights using an optimizer

Model checkpoints are saved during training.

---

##  5. Initial Evaluation

After training, the model is evaluated on the validation dataset to obtain baseline performance metrics such as:

* Precision
* Recall
* Detection confidence scores

These metrics help assess how well the model identifies viable and non-viable seeds.

---

##  6. Confidence Threshold Optimization

A **random search over detection confidence thresholds** is performed on the validation dataset.

This step helps determine the optimal threshold that balances:

* **Precision** (reducing false positives)
* **Recall** (detecting more seeds)

The threshold that produces the best performance is selected for final evaluation.

---

##  7. Recall Improvement Training

To further improve detection performance, the model is retrained for additional epochs starting from the previously trained weights.

This step helps improve **recall**, allowing the model to detect harder or previously missed seed instances.

---

##  8. Final Model Evaluation

The optimized model is evaluated again using the tuned confidence threshold.

Final evaluation provides the performance of the model on unseen images and validates its effectiveness for seed viability detection.

---

##  Pipeline Summary

```
Dataset Preparation
        ↓
DataLoader Construction
        ↓
Model Initialization (Faster R-CNN)
        ↓
Model Training
        ↓
Initial Evaluation
        ↓
Confidence Threshold Optimization
        ↓
Recall Improvement Training
        ↓
Final Model Evaluation
```

---


