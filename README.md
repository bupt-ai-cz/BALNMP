# Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides ![visitors](https://visitor-badge.glitch.me/badge?page_id=bupt-ai-cz.BALNMP)
[Arxiv](https://arxiv.org/abs/2112.02222) | [Project](https://bupt-ai-cz.github.io/BALNMP/) | [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"Predicting%20Axillary%20Lymph%20Node%20Metastasis%20in%20Early%20Breast%20Cancer"%20&url=https://github.com/bupt-ai-cz/BALNMP)

This repo is the official implementation and dataset introduction of our paper "Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides".

Our paper is accepted by [Frontiers in Oncology](https://www.frontiersin.org/articles/10.3389/fonc.2021.759007/full), and you can also get access our paper from [MedRxiv](https://www.medrxiv.org/content/10.1101/2021.10.10.21264721) or [Arxiv](https://arxiv.org/abs/2112.02222).

## Abstract

- Objectives: To develop and validate a deep learning (DL)-based primary tumor biopsy signature for predicting axillary lymph node (ALN) metastasis preoperatively in early breast cancer (EBC) patients with clinically negative ALN.

- Methods: A total of 1,058 EBC patients with pathologically confirmed ALN status were enrolled from May 2010 to August 2020. A DL core-needle biopsy (DL-CNB) model was built on the attention-based multiple instance-learning (AMIL) framework to predict ALN status utilizing the DL features, which were extracted from the cancer areas of digitized whole-slide images (WSIs) of breast CNB specimens annotated by two pathologists. Accuracy, sensitivity, specificity, receiver operating characteristic (ROC) curves, and areas under the ROC curve (AUCs) were analyzed to evaluate our model.

- Results: The best-performing DL-CNB model with VGG16_BN as the feature extractor achieved an AUC of 0.816 (95% confidence interval (CI): 0.758, 0.865) in predicting positive ALN metastasis in the independent test cohort. Furthermore, our model incorporating the clinical data, which was called DL-CNB+C, yielded the best accuracy of 0.831 (95% CI: 0.775, 0.878), especially for patients younger than 50 years (AUC: 0.918, 95% CI: 0.825, 0.971). The interpretation of DL-CNB model showed that the top signatures most predictive of ALN metastasis were characterized by the nucleus features including density (*p* = 0.015), circumference (*p* = 0.009), circularity (*p* = 0.010), and orientation (*p* = 0.012).

- Conclusion: Our study provides a novel DL-based biomarker on primary tumor CNB slides to predict the metastatic status of ALN preoperatively for patients with EBC.

## Results

We have recomputed the classification results with argmax prediction rule, and they are different with the results in our [paper](https://arxiv.org/abs/2112.02222), which are computed based on the [cut-off value in ROC](https://en.wikipedia.org/wiki/Youden%27s_J_statistic#:~:text=Youden%27s%20index%20is,as%20informedness.%5B3%5D). The dataset files for each experiments setting are placed [here](https://github.com/bupt-ai-cz/BALNMP/tree/main/dataset_json).

### The performance in prediction of ALN status (N0 vs. N(+))

<div class="tg-wrap" align="center"><table><thead><tr><th>Methods</th><th></th><th>AUC</th><th>Accuracy (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1-score (%)</th></tr></thead><tbody><tr><td rowspan="3">Clinical data only</td><td>T</td><td>0.661 </td><td>64.13 </td><td>52.46 </td><td>62.08 </td><td>56.87 </td></tr><tr><td>V</td><td>0.709 </td><td>68.10 </td><td>56.52 </td><td>65.82 </td><td>60.82 </td></tr><tr><td>I-T</td><td>0.613 </td><td>57.80 </td><td>45.92 </td><td>53.57 </td><td>49.45 </td></tr><tr><td rowspan="3">DL-CNB model</td><td>T</td><td>0.901 </td><td>81.43 </td><td>72.53 </td><td>82.50 </td><td>77.19 </td></tr><tr><td>V</td><td>0.808 </td><td>71.43 </td><td>66.10 </td><td>49.37 </td><td>56.52 </td></tr><tr><td>I-T</td><td>0.816 </td><td>70.18 </td><td>67.27 </td><td>44.05 </td><td>53.24 </td></tr><tr><td rowspan="3">DL-CNB+C model</td><td>T</td><td>0.878 </td><td>78.89 </td><td>70.82 </td><td>75.83 </td><td>73.24 </td></tr><tr><td>V</td><td>0.823 </td><td>74.76 </td><td>64.13 </td><td>74.68 </td><td>69.01 </td></tr><tr><td>I-T</td><td>0.831 </td><td>76.15 </td><td>70.51 </td><td>65.48 </td><td>67.90 </td></tr></tbody></table></div>

### The performance in prediction of ALN status (N0 vs. N + (1-2))

<div class="tg-wrap" align="center"><table><thead><tr><th>Methods</th><th>　</th><th>AUC</th><th>Accuracy (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1-score (%)</th></tr></thead><tbody><tr><td rowspan="3">Clinical&nbsp;&nbsp;&nbsp;data only</td><td>T</td><td>0.638 </td><td>61.58 </td><td>34.08 </td><td>59.38 </td><td>43.30 </td></tr><tr><td>V</td><td>0.677 </td><td>64.57 </td><td>36.76 </td><td>56.82 </td><td>44.64 </td></tr><tr><td>I-T</td><td>0.627 </td><td>61.05 </td><td>29.58 </td><td>55.26 </td><td>38.53 </td></tr><tr><td rowspan="3">DL-CNB model</td><td>T</td><td>0.912 </td><td>81.66 </td><td>66.67 </td><td>51.56 </td><td>58.15 </td></tr><tr><td>V</td><td>0.756 </td><td>70.86 </td><td>42.86 </td><td>47.73 </td><td>45.16 </td></tr><tr><td>I-T</td><td>0.845 </td><td>80.23 </td><td>54.55 </td><td>63.16 </td><td>58.54 </td></tr><tr><td rowspan="3">DL-CNB+C model</td><td>T</td><td>0.936 </td><td>85.71 </td><td>75.00 </td><td>63.28 </td><td>68.64 </td></tr><tr><td>V</td><td>0.789 </td><td>77.14 </td><td>61.11 </td><td>25.00 </td><td>35.48 </td></tr><tr><td>I-T</td><td>0.878 </td><td>84.88 </td><td>77.27 </td><td>44.74 </td><td>56.67 </td></tr></tbody></table></div>

### The performance in prediction of ALN status (N0 vs. N + (≥3))

<div class="tg-wrap" align="center"><table><thead><tr><th>Methods</th><th>　</th><th>AUC</th><th>Accuracy (%)</th><th>Precision (%)</th><th>Recall (%)</th><th>F1-score (%)</th></tr></thead><tbody><tr><td rowspan="3">Clinical&nbsp;&nbsp;&nbsp;data only</td><td>T</td><td>0.680 </td><td>66.86 </td><td>38.05 </td><td>65.00 </td><td>48.00 </td></tr><tr><td>V</td><td>0.748 </td><td>68.48 </td><td>37.14 </td><td>76.47 </td><td>50.00 </td></tr><tr><td>I-T</td><td>0.629 </td><td>60.12 </td><td>30.26 </td><td>58.97 </td><td>40.00 </td></tr><tr><td rowspan="3">DL-CNB model</td><td>T</td><td>0.906 </td><td>82.94 </td><td>64.10 </td><td>62.50 </td><td>63.29 </td></tr><tr><td>V</td><td>0.755 </td><td>76.97 </td><td>38.89 </td><td>20.59 </td><td>26.92 </td></tr><tr><td>I-T</td><td>0.837 </td><td>80.92 </td><td>71.43 </td><td>25.64 </td><td>37.74 </td></tr><tr><td rowspan="3">DL-CNB+C model</td><td>T</td><td>0.918 </td><td>81.96 </td><td>68.92 </td><td>42.50 </td><td>52.58 </td></tr><tr><td>V</td><td>0.761 </td><td>79.39 </td><td>50.00 </td><td>29.41 </td><td>37.04 </td></tr><tr><td>I-T</td><td>0.838 </td><td>80.35 </td><td>66.67 </td><td>25.64 </td><td>37.04 </td></tr></tbody></table></div>

## Pre-Trained Models

Please download pre-trained models from [here](https://drive.google.com/drive/folders/1W7kBL_kdzFuPS5jvI-liHCIe6YVl505z?usp=sharing).

## Demo Software

We have also provided software for easily checking the performance of our model to predict ALN metastasis.

Please download the software from [here](https://drive.google.com/drive/folders/1ItKCldu8vbHhbZvhXic-11Ei-NVGBZU2?usp=sharing), and check the `README.txt` for usage. Please note that this software is only used for demo, and it cannot be used for other purposes.

<div align="center">
    <img src="imgs/demo-software.png" alt="demo-software" height="25%" width="25%" />
</div>

# BCNB Dataset

Our paper has introduced a new dataset of **Early Breast Cancer Core-Needle Biopsy WSI (BCNB)**, which includes core-needle biopsy whole slide images (WSIs) of early breast cancer patients and the corresponding clinical data. Based on this dataset, we have studied the deep learning algorithm for predicting the metastatic status of axillary lymph node (ALN) preoperatively by using multiple instance learning (MIL), and have achieved the best AUC of 0.831 in the independent test cohort. For more details, please review our [paper](https://arxiv.org/abs/2112.02222). 

For full access to the BCNB dataset, please visit the [project page](https://bupt-ai-cz.github.io/BALNMP/).

## Description

There are a totally of **1058 patients** and they are divided into the following 3 categories according to the axillary lymph node (ALN) metastasis:

- N0: having no positive lymph nodes (655 patients, 61.9%).
- N+(1-2): having one or two positive lymph nodes (210 patients, 19.8%).
- N+(>2): having three or more positive lymph nodes (193 patients, 18.3%).

**Part of tumor regions are annotated in WSIs, the extra annotations should be done by yourself if needed. Except for the WSIs, we have also provided the clinical data of each patient, which includes age, tumor size, tumor type, ER, PR, HER2, HER2 expression, histological grading, surgical, Ki67, molecular subtype, number of lymph node metastases, label.**

The WSIs are provided with `.jpg` format and the clinical data are provided with `.xlsx` format. The dataset is collected and organized by the experienced doctors of our research group.

Based on this dataset, we have studied the prediction of the metastatic status of axillary lymph node (ALN) in our [paper](https://arxiv.org/abs/2112.02222), which is a weakly supervised classification task. However, other researches based on our dataset are also feasible, such as the prediction of histological grading, molecular subtype, HER2, ER, and PR. We do not limit the specific content for your research, and any research based on our dataset is welcome.

**Please note that the dataset is only used for education and research, and the usage for commercial and clinical applications is not allowed. The usage of this dataset must follow the [license](https://github.com/bupt-ai-cz/BALNMP#license).** 

## Annotation

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

## Code for data preprocessing

We provided some codes for data preprocessing, which can be used to extract annotated tumor regions of all WSIs, and cutting patches with fixed size from all extracted annotated tumor regions, they may be helpful for you. Please check the [code](https://github.com/bupt-ai-cz/BALNMP/tree/main/code) for more details.

## Example

Here we have provided some WSIs and clinical data. For full access to the dataset, please visit the [project page](https://bupt-ai-cz.github.io/BALNMP/).

### WSI

#### N0

<div align="center">
    <img src="imgs/N0.png" alt="N0" height="70%" width="70%" />
</div>

#### N+(1-2)
<div align="center">
    <img src="imgs/N+(1~2).png" alt="N+(1-2)" height="70%" width="70%" />
</div>

#### N+(>2)
<div align="center">
    <img src="imgs/N+(%EF%BC%9E2).png" alt="N+(>2)" height="50%" width="50%" />
</div>

### Clinical Data

<div align="center">
    <img src="imgs/clinical-data.png" alt="clinical-data" />
</div>

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

This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to the dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the website.
2. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).
3. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
4. That all rights not expressly granted to you are reserved by us.

## Contact

If you encounter any problems please contact us without hesitation.

- email: tangwenqi@bupt.edu.cn, czhu@bupt.edu.cn, drxufeng@mail.ccmu.edu.cn
