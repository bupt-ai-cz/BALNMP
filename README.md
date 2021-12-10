# Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides ![visitors](https://visitor-badge.glitch.me/badge?page_id=bupt-ai-cz.BALNMP)
[Arxiv](https://arxiv.org/abs/2112.02222) | [Project](https://bupt-ai-cz.github.io/BALNMP/) | [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"Predicting%20Axillary%20Lymph%20Node%20Metastasis%20in%20Early%20Breast%20Cancer"%20&url=https://github.com/bupt-ai-cz/BALNMP)

This repo is the official implementation and dataset introduction of our paper "Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides".

Our paper is accepted by [Frontiers in Oncology](https://www.frontiersin.org/articles/10.3389/fonc.2021.759007/full), and you can also get access our paper from [MedRxiv](https://www.medrxiv.org/content/10.1101/2021.10.10.21264721) or [Arxiv](https://arxiv.org/abs/2112.02222).

## Abstract

- Objectives: To develop and validate a deep learning (DL)-based primary tumor biopsy signature for predicting axillary lymph node (ALN) metastasis preoperatively in early breast cancer (EBC) patients with clinically negative ALN.

- Methods: A total of 1,058 EBC patients with pathologically confirmed ALN status were enrolled from May 2010 to August 2020. A DL core-needle biopsy (DL-CNB) model was built on the attention-based multiple instance-learning (AMIL) framework to predict ALN status utilizing the DL features, which were extracted from the cancer areas of digitized whole-slide images (WSIs) of breast CNB specimens annotated by two pathologists. Accuracy, sensitivity, specificity, receiver operating characteristic (ROC) curves, and areas under the ROC curve (AUCs) were analyzed to evaluate our model.

- Results: The best-performing DL-CNB model with VGG16_BN as the feature extractor achieved an AUC of 0.816 (95% confidence interval (CI): 0.758, 0.865) in predicting positive ALN metastasis in the independent test cohort. Furthermore, our model incorporating the clinical data, which was called DL-CNB+C, yielded the best accuracy of 0.831 (95% CI: 0.775, 0.878), especially for patients younger than 50 years (AUC: 0.918, 95% CI: 0.825, 0.971). The interpretation of DL-CNB model showed that the top signatures most predictive of ALN metastasis were characterized by the nucleus features including density (*p* = 0.015), circumference (*p* = 0.009), circularity (*p* = 0.010), and orientation (*p* = 0.012).

- Conclusion: Our study provides a novel DL-based biomarker on primary tumor CNB slides to predict the metastatic status of ALN preoperatively for patients with EBC.

## Dataset

Our paper has introduced a new dataset that includes whole slide images (WSIs) of early breast cancer patients and the corresponding clinical data. Based on this dataset, we have studied the deep learning algorithm for predicting the metastatic status of axillary lymph node (ALN) preoperatively by using multiple instance learning (MIL), and have achieved the best AUC of 0.831 in the independent test cohort. For more details, please review our [paper](https://arxiv.org/abs/2112.02222).

### Description

The dataset is collected and organized by the experienced doctors of our research group.

There are a totally of 1058 patients and they are divided into the following 3 categories according to the axillary lymph node (ALN) metastasis:

- N0: having no positive lymph nodes (655 patients, 61.9%).
- N+(1-2): having one or two positive lymph nodes (210 patients, 19.8%).
- N+(>2): having three or more positive lymph nodes (193 patients, 18.3%).

Except for the WSIs, we have also provided the clinical data of each patient, which includes age, tumor size, tumor type, ER, PR, HER2, HER2 expression, histological grading, surgical, Ki67, molecular subtype, number of lymph node metastases, label.

The WSIs is provided with `.jpg` format and the clinical data is provided with `.xlsx` format.

The usage of this dataset must follow the [license](https://github.com/bupt-ai-cz/BALNMP#license). We have used this dataset for weakly supervised classification, and we do not limit the specific task types for your research. Please note that the dataset is only used for education and research, and the usage for commercial and clinical applications is not allowed.

### Annotation

Annotation information is stored in `.json` with the following format, where `vertices` have recorded coordinates of each point in the polygonal annotated area.

```json
{
    "positive": [
        {
            "name": "Annotation 0",
            "vertices": [
                [
                    14274,
                    10723
                ],
                [
                    14259,
                    10657
                ],
                ......
            ]
        }
    ],
    "negative": []
}
```

### Code for preprocessing

We have used the codes for extracting annotated tumor regions of all WSIs and cutting patches with fixed size from all extracted annotated tumor regions, and they may be helpful for your research. Please check the [code](https://github.com/bupt-ai-cz/BALNMP/tree/main/code) for more details.

### Example

Here we have provided some WSIs and clinical data. For full access to the dataset, please contact us and the usage must follow the [license](https://github.com/bupt-ai-cz/BALNMP#license).

#### WSI

##### N0

<img src="imgs/N0.png" alt="N0" style="zoom: 75%;" />

##### N+(1-2)

<img src="imgs/N+(1~2).png" alt="N+(1~2)" style="zoom:75%;" />

##### N+(>2)

<img src="imgs/N+(%EF%BC%9E2).png" alt="N+(＞2)" style="zoom:75%;" />

#### Clinical Data

![](imgs/clinical-data.png)


## Pre-Trained Models

Please download pre-trained models from [here](https://drive.google.com/drive/folders/1W7kBL_kdzFuPS5jvI-liHCIe6YVl505z?usp=sharing).

## Demo Software

We have also provided software for easily checking the performance of our model to predict ALN metastasis.

Please download the software from [here](https://drive.google.com/drive/folders/1ItKCldu8vbHhbZvhXic-11Ei-NVGBZU2?usp=sharing), and check the `README.txt` for usage. Please note that this software is only used for demo, and it cannot be used for other purposes.

![demo-software](imgs/demo-software.png)

## Citation

Please cite our paper in your publications if it helps your research.

```
@article{xu2021predicting,
  title={Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides},
  author={Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan},
  journal={Frontiers in Oncology},
  pages={4133},
  year={2021},
  publisher={Frontiers}
}
```

## License

This BALNMP Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to the BALNMP Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the BALNMP website.
2. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).
3. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
4. That all rights not expressly granted to you are reserved by us.

## Contact

If you encounter any problems please contact us without hesitation.

- email: tangwenqi@bupt.edu.cn, czhu@bupt.edu.cn, drxufeng@mail.ccmu.edu.cn
