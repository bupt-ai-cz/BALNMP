# Multi-Task Learning with Attention-Based Multiple Instance Learning for Breast Cancer Metastasis Prediction

###### contributor @Lan Lan

### backbone_builder.py
Implements the feature extractor module.

### attention_aggregator.py
Attention module, as used in baseline work, see citation in file.

### mil_net.py
Model Architecture - contains the baseline, multi-task and single-task model.

### train_multitask.py
Train multi-task models

### train_singletask.py
Train single-task models

### CS230_training_logs.py
Contains each configuration and logs for training.

### code/dataset
- clinical_data folder: pre-processed clinical data.
- json folder: contains the path to each patches in bags

### update_status_data.ipynb
- update the old json files in code/dataset/json to include both metastasis label (binary) and severity (multi-class: 0, 1, 2).


