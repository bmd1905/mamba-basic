# Simple Mamba Classifier

This is a simple Mamba Classifier using a Mamba model implemented in [@alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py).

## Usage

### Install dependencies

```bash
conda create -n mamba-classifier python=3.10 -y
conda activate mamba-classifier
```

```bash
pip install -r requirements.txt
```

If you want to use `wandb` to track the experiment, you can set up an account and run:

```bash
wandb login
```

### Train the model

```bash
python3 train.py
```
