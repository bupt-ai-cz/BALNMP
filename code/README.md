## Setup

### Environment

Create environment and install dependencies.

```bash
conda create -n BALNMP python=3.6 -y
conda activate BALNMP
pip install -r requirements.txt
```

### Dataset

For your convenience, we have provided preprocessed WSI patches and corresponding clinical data.

Please download the dataset from [here](https://drive.google.com/file/d/1KKbdsmCaA4xKDdOPdTXuxIga4o9ZUrLG/view?usp=sharing), and unzip them by the following scripts:

```bash
cd code
mkdir dataset
unzip paper_dataset.zip -d dataset
```

## Training

Our codes have supported the following experiments, whose results have been presented in our [paper and supplementary material](https://arxiv.org/abs/2112.02222).

> experiment_index:
> 
> 0. N0 vs N+(>0)
> 1. N+(1-2) vs N+(>2)
> 2. N0 vs N+(1-2) vs N+(>2)
> 3. N0 vs N+(1-2)
> 4. N0 vs N+(>2)

To run any experiment, you can do as this:

```bash
bash run.sh ${experiment_index}
```

Furthermore, if you want to try other settings, please see `train.py` for more details.