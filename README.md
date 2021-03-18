# TargetNet: Functional microRNA Target Prediction with Deep Neural Networks

## Abstract
<p style="text-align:justify">
<strong>Motivation:</strong> MicroRNAs (miRNAs) play pivotal roles in gene expression regulation by binding to target sites of messenger RNAs (mRNAs). While identifying functional targets of miRNAs is of utmost importance, their prediction remains a great challenge. Previous computational algorithms have major limitations. They use conservative candidate target site (CTS) selection criteria mainly focusing on canonical site types, rely on laborious and time-consuming manual feature extraction, and do not fully capitalize on the information underlying miRNA-CTS interactions.
<br/>
<strong>Results:</strong> In this paper, we introduce TargetNet, a novel deep learning-based algorithm for functional miRNA target prediction. To address the limitations of previous approaches, TargetNet has three key components: (1) relaxed CTS selection criteria accommodating irregularities in the seed region, (2) a novel miRNA-CTS sequence encoding scheme incorporating extended seed region alignments, and (3) a deep residual network-based prediction model. The proposed model was trained with miRNA-CTS pair datasets and evaluated with miRNA-mRNA pair datasets. TargetNet advances the previous state-of-the-art algorithms used in functional miRNA target classification. Furthermore, it demonstrates great potential for distinguishing high-functional miRNA targets.
<br/><br/>
</p>
<br/>

## Installation
We recommend creating a Miniconda environment from the <code>TargetNet.yaml</code> file as:
```
conda env create -f TargetNet.yaml
```
Alternatively, you can install the necessary python packages from the <code>requirements.txt</code> file as:
```
pip install -r requirements.txt 
```
<br/>

## Data Format
Extract the <code>data.tar.gz</code> file to obtain the dataset used for the manuscript as
```
tar -zxvf data.tar.gz
```
If you want to use other datasets, please follow the data format described as follows

- The dataset must be in a tab-delimited file with **at least 4 columns**
- The first row must be a header line (thus, will not be processed by the TargetNet algorithm).
- The 1st ~ 4th columns must hold the following information
    - [1st column] miRNA id
    - [2nd column] miRNA sequence
    - [3rd column] mRNA id
    - [4th column] mRNA sequence -- mRNA sequence must be longer than 40 nucleotides
- For the file containing train and validation datasets, it requires **additional 2 columns**
    - [5th column] label -- *0* or *1*
    - [6th column] split -- *train* or *val*
    
Please refer to the provided dataset files for more detailed examples.
<br/><br/>

## How to Run
### Training a TargetNet model
You can use the <code>train_model.py</code> script with the necessary configuration files as
```
CUDA_VISIBLE_DEVICES=0 python train_model.py --data-config config/data/miRAW_train.json --model-config config/model/TargetNet.json --run-config config/run/run.json --output-path results/TargetNet_training/
```
The script will generate a <code>TargetNet.pt</code> file containing a trained model. <br>
For using other datasets, modify the data paths specified in the <code>miRAW_train.json</code> data-config file.

### Evaluating a TargetNet model
You can use the <code>evaluate_model.py</code> script with the necessary configuration files as
```
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --data-config config/data/miRAW_eval.json --model-config config/model/TargetNet.json --run-config config/run/run.json --checkpoint pretrained_models/TargetNet.pt --output-path results/TargetNet-evaluation/
```
The script will generate a tab-delimited <code>*_outputs.txt</code> file described as follows

- The output file contains a TargetNet prediction for each miRNA-mRNA set from the given dataset.
    - [1st column] set_idx -- indicates the set index from the given dataset.
    - [2nd column] output -- TargetNet prediction score ranging from 0 to 1.
- For binary classification results, use a threshold of 0.5 to binarize the output scores.

For using other datasets, modify the data paths specified in the <code>miRAW_eval.json</code> data-config file.
<br/><br/><br/>
