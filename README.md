# TargetNet: Functional microRNA Target Prediction with Deep Neural Networks

## Abstract
<p style="text-align:justify">
<strong>Motivation:</strong> MicroRNAs (miRNAs) play pivotal roles in gene expression regulation by binding to target sites of messenger RNAs (mRNAs). While identifying functional targets of miRNAs is of utmost importance, their prediction remains a great challenge. Previous computational algorithms have major limitations. They use conservative candidate target site (CTS) selection criteria mainly focusing on canonical site types, rely on laborious and time-consuming manual feature extraction, and do not fully capitalize on the information underlying miRNA-CTS interactions.
<br/>
<strong>Results:</strong> In this paper, we introduce TargetNet, a novel deep learning-based algorithm for functional miRNA target prediction. To address the limitations of previous approaches, TargetNet has three key components: (1) relaxed CTS selection criteria accommodating irregularities in the seed region, (2) a novel miRNA-CTS sequence encoding scheme incorporating extended seed region alignments, and (3) a deep residual network-based prediction model. The proposed model was trained with miRNA-CTS pair datasets and evaluated with miRNA-mRNA pair datasets. TargetNet advances the previous state-of-the-art algorithms used in functional miRNA target classification. Furthermore, it demonstrates great potential for distinguishing high-functional miRNA targets.
<br/><br/>
</p>
<br/>

## How to Run
#### Example:
```
CUDA_VISIBLE_DEVICES=0 python train_model.py --data-config config/data/miRAW_train.json --model-config config/model/TargetNet.json --run-config config/run/run.json --output-path results/
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --data-config config/data/miRAW_eval.json --model-config config/model/TargetNet.json --run-config config/run/run.json --checkpoint pretrained_models/TargetNet.pt --output-path results/
```
<br/>

## Data
- <a href="http://ailab.snu.ac.kr/TargetNet/miRAW_Train_Validation.tar.gz">miRAW Train & Validation</a>
- <a href="http://ailab.snu.ac.kr/TargetNet/miRAW_Test.tar.gz">miRAW Test</a>
- <a href="http://ailab.snu.ac.kr/TargetNet/LFC_Test.tar.gz">LFC Test</a>
<br/>

## Requirements
- Python 3.8
- PyTorch 1.5
- Numpy 1.19.1
- Pandas 1.1.1
<br/><br/><br/>
