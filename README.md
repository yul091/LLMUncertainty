# LLMUncertainty
Uncertainty Awareness of Large Language Models Under Code Distribution Shifts: A Benchmark Study


## Requirements
- [python3.7+](https://www.python.org/downloads/release/python-380/)
- [PyTorch 1.13.0](https://pytorch.org/get-started/locally/)
- Libraries and dependencies:
```
pip install -r requirements.txt
```

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/yul091/LLMUncertainty.git
cd LLMUncertainty
```
### Step 1: Download the preprocessed Java-small dataset (~60 K examples, compressed: 84MB) and Python150k dataset for OOD detection (~150 K examples, compressed: 526MB)
```
wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz
tar -xvzf java-small.tar.gz
wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
tar -xzvf py150.tar.gz
```

### Step 2: Training a model
#### Training a model from scratch
To train a model from scratch:
- Edit the file [scripts/train_cs.sh](scripts/train_cs.sh) and file [scripts/train_cc.sh](scripts/train_cc.sh) to point to the right preprocessed data and a specific model archiecture.
- Before training, you can edit the configuration hyper-parameters in these two files.
- Run the two shell scripts:
```
bash scripts/train_cs.sh # code summary
bash scripts/train_cc.sh # code completion
```
### Step 3: Running the five probabilistic methods for calibration and UE
- Edit the file [scripts/get_uncertainty_cs.sh](scripts/get_uncertainty_cs.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [scripts/get_uncertainty_cs.sh](scripts/get_uncertainty_cs.sh):
```
bash scripts/get_uncertainty_cs.sh # code summary
bash scripts/get_uncertainty_cc.sh # code completion
```
### Step 4: Evaluation the UE quality in misclassification detection
- Edit the script [scripts/misclassification_prediction_cs.sh](scripts/misclassification_prediction_cs.sh) to point to the target evaluation choice (misclassification detection or OOD detection).
- Run the script [scripts/misclassification_prediction_cs.sh](scripts/misclassification_prediction_cs.sh):
```
scripts/misclassification_prediction_cs.sh # code summary
scripts/misclassification_prediction_cc.sh # code completion
```
### Step 5: Evaluation the UE quality in selective prediction via abstention
- Edit the script [scripts/abstention_cs.sh](scripts/abstention_cs.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [scripts/abstention_cs.sh](scripts/abstention_cs.sh):
```
bash scripts/abstention_cs.sh # code summary
bash scripts/abstention_cc.sh # code completion
```
### Step 6: Evaluation the UE quality in OOD detection
- Edit the script [scripts/ood_detection_cs.sh](scripts/ood_detection_cs.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [scripts/ood_detection_cs.sh](scripts/ood_detection_cs.sh):
```
bash scripts/ood_detection_cs.sh # code summary
bash scripts/ood_detection_cc.sh # code completion
```