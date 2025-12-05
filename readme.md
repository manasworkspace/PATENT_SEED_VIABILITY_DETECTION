


---

#  Patent

This project is associated with an officially published Indian patent describing the proposed AI-based seed viability detection framework.

## Patent Information

| Field                            | Details                                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------------------- |
| **Title**                        | *An AI-Powered Computer Vision Approach for Seed Viability Detection in Hydroponic Systems* |
| **Application Number**           | **202511097814 A**                                                                          |
| **Country**                      | India                                                                                       |
| **Filing Date**                  | 10 October 2025                                                                             |
| **Publication Date**             | 05 December 2025                                                                            |
| **Applicant**                    | Manipal University Jaipur                                                                   |
| **International Classification** | G06N0003045000, G06N0003080000, G06N0020000000, G06N0003098000, G06N0003096000              |
| **Pages**                        | 13                                                                                          |
| **Claims**                       | 10                                                                                          |

---

## Inventors

* Dr. Hemlata Parmar
* Parth Gupta
* Singh Manas Mukundkumar
* Abhinav Mukherjee
* Dr. Utsav Krishan Murari

---

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

Here is your **cleaned and properly formatted GitHub README section** with correct Markdown headings, tables, and entity references. You can paste this directly into `README.md`.

---

#  Experimental Results

This section summarizes the experimental evaluation of the seed viability detection model trained using **Faster R-CNN** with a **ResNet-50 FPN** backbone implemented in **PyTorch** and **Torchvision**.

The dataset annotations follow the **COCO format**, and evaluation is performed using **pycocotools**.

---

# 1️ Initial Model Evaluation

The baseline model was trained on approximately **5,000 seed images** using COCO-style bounding box annotations.

Evaluation was first performed using the **default detection confidence threshold of 0.5** on the validation dataset.

### Baseline Metrics

| Metric    | Score      |
| --------- | ---------- |
| Precision | **0.5984** |
| Recall    | **0.9976** |
| F1 Score  | **0.7481** |

### Interpretation

The baseline model demonstrates:

* **Very high recall (~1.0)**
* **Moderate precision**

This indicates that the model successfully detects nearly all seeds but also produces a relatively large number of **false positives**.

To improve the balance between precision and recall, **confidence threshold optimization** was performed.

---

# 2️ Threshold Optimization (Random Search)

Object detection models often require tuning the **confidence threshold** used to accept predicted bounding boxes.

A **random search experiment** was conducted across thresholds from **0.10 to 0.80** to determine the value that maximizes the **F1 score** on the validation dataset.

### Validation Results

| Threshold | Precision | Recall | F1 Score   |
| --------- | --------- | ------ | ---------- |
| 0.10      | 0.9935    | 0.3937 | **0.5639** |
| 0.20      | 0.9956    | 0.3488 | 0.5166     |
| 0.30      | 0.9970    | 0.3262 | 0.4916     |
| 0.40      | 0.9987    | 0.3112 | 0.4745     |
| 0.50      | 0.9988    | 0.3005 | 0.4620     |
| 0.60      | 0.9988    | 0.2906 | 0.4502     |
| 0.70      | 0.9990    | 0.2838 | 0.4421     |
| 0.80      | 0.9992    | 0.2763 | 0.4329     |

### Best Threshold

| Metric         | Value      |
| -------------- | ---------- |
| Best Threshold | **0.10**   |
| Best F1 Score  | **0.5639** |

### Interpretation

Lower thresholds allow the model to accept more detections, which increases **recall** but may introduce additional **false positives**.

The threshold **0.10** provided the best balance between precision and recall and was selected for **final evaluation**.

---

# 3️ Final Test Set Evaluation

The optimized model was evaluated on the **held-out test dataset** using the best threshold discovered during validation.

### Final Detection Metrics

| Metric    | Score      |
| --------- | ---------- |
| Precision | **0.9937** |
| Recall    | **0.3773** |
| F1 Score  | **0.5470** |

### Interpretation

The final model demonstrates:

* **Very high precision (~99%)**
* **Moderate recall (~37%)**

This indicates that the model produces **highly reliable detections with very few false positives**, though some seeds may remain undetected.

---

# 4️ COCO Evaluation Metrics

To provide standardized object detection evaluation, the model was also evaluated using the **COCO Mean Average Precision (mAP)** metric.

### COCO mAP Results

| Metric              | Score     |
| ------------------- | --------- |
| mAP (IoU 0.50–0.95) | **0.465** |
| mAP (IoU 0.50)      | **0.493** |
| mAP (IoU 0.75)      | **0.493** |

### Average Recall

| Metric              | Score     |
| ------------------- | --------- |
| AR @ 1 Detection    | **0.099** |
| AR @ 10 Detections  | **0.481** |
| AR @ 100 Detections | **0.481** |

These metrics follow the official **COCO evaluation protocol** implemented through **pycocotools**.

---

# 5️ Summary

The proposed **Seed Viability Detection Pipeline** demonstrates the effectiveness of deep learning–based object detection for agricultural seed analysis.

Key achievements include:

* **High precision seed detection (~99%)**
* **Robust object localization**
* **COCO-standardized evaluation metrics**
* **Optimized detection thresholds**

The results highlight the potential of **Faster R-CNN** for automated seed viability analysis in agricultural applications.

---



