<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis </h1>

<p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> [<a href="https://arxiv.org/abs/2605.09664" target="_blank">ArXiv Paper Link</a>] </p>

<p align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <b> Zahra Asadi*, Haeseung Jeon*, Sohyun Han, Md Mahmuduzzaman Kamol, Se Eun Oh, Mohammad Saidur Rahman† </b> </p>

<p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> *Equally credited authors.  †Corresponding author. </p>


> [!NOTE]
> This is official implementation of the paper *FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis*.


## FreeMOCA Pipeline

<img width="5200" height="1400" alt="FreeMOCA_Overview_v3" src="https://github.com/user-attachments/assets/93d44b2d-7f16-4002-8c9c-68ca8274ecac" />

FreeMOCA operates in the following process:

1. Tasks arrive sequentially, each introducing new malware families.
2. The model for each task is initialized from the previous task (**warm-start**) to align task minima in parameter space.
3. After training, a **mode connectivity–based interpolation** is computed between the current and previous task optima. This process can use **adaptive layer-wise interpolation** based on parameter update magnitude.
4. The interpolated model replaces the current model and is used for the next task.
5. No past data, features, or generated samples are stored, resulting in **memory-free continual learning**.

The entire process constructs a chain of connected solutions that lie on a low-loss manifold, significantly reducing catastrophic forgetting.

## Datasets

FreeMOCA was evaluated with two large-scale malware datasets, EMBER and Androzoo. Dataset sources:
- EMBER: https://github.com/elastic/ember  
- Androzoo (AZ): https://zenodo.org/records/14537891

Download and set up the dataset in the following directory:

```python
FreeMOCA/data/
├── AZ_Class
│   ├── AZ_Class_Test.npz
│   └── AZ_Class_Train.npz
│
├── AZ_Domain
│   ├── 2008_Domain_AZ_Test_Transfo...
│   ├── 2008_Domain_AZ_Train_Transfo...
│   ├── 2009_Domain_AZ_Test_Transfo...
│   └── ...
│
├── EMBER_Class
│   ├── XY_test.npz
│   └── XY_train.npz
│
└── EMBER_Domain
    ├── 2018-01
    ├── 2018-02
    ├── 2018-03
    └── ...
```

## Running FreeMOCA

This repository supports two continual learning scenarios:

- **Class-Incremental Learning (Class-IL)**: new classes (malware families) are introduced over tasks.
- **Domain-Incremental Learning (Domain-IL)**: label space is fixed, but data distribution shifts across time (e.g., months/years).

### Step 1. Installation

Run following command for the 

```bash
conda create -n freemoca python=3.9
conda activate freemoca
pip install -r requirements.txt

```


### Step 2. Run FreeMOCA (Class-IL & Domain-IL)

```python
# EMBER Class-IL
cd ./FreeMOCA_Class/EMBER_Class
CUDA_VISIBLE_DEVICES=0 python main.py --train_data /path/to/data --test_data /path/to/data
```

```python
# AZ Class-IL
cd ./FreeMOCA_Class/AZ_Class
CUDA_VISIBLE_DEVICES=0 python main.py --train_data /path/to/data --test_data /path/to/data
```

```python
# EMBER Domain-IL
cd ./FreeMOCA_Domain/EMBER_Domain
CUDA_VISIBLE_DEVICES=0 python main.py --data_root /path/to/data/directory
```

```python
# AZ Domain-IL
cd ./FreeMOCA_Domain/AZ_Domain
CUDA_VISIBLE_DEVICES=0 python main.py --data_root /path/to/data/directory
```

For a more detailed setup in hyperparameters, check up **Appendix A. Common Arguments for FreeMOCA**.


###  Appendix A. Common Arguments for FreeMOCA

To adjust the hyperparameters or experimental settings, use the following arguments:

| Argument | Description |
|-----------|-------------|
| `--init_classes` | Number of classes at task 0 |
| `--epochs` | Epochs per task |
| `--batchsize` | Batch size |
| `--lr` | Learning rate |
| `--momentum` | SGD momentum |
| `--weight_decay` | Weight decay |
| `--lambda_min` | Minimum interpolation weight |
| `--lambda_max` | Maximum interpolation weight |

To change the default setting of arguments, check the `arguments.py` file.

## Baselines

Our repository supports experiments on the following standard baselines:

- **None**: Train only on the current task (lower bound)
- **Joint**: Train on all data seen so far (upper bound, impractical)

You can run it with the following command:

```python
cd /path/to/experiment/directory
CUDA_VISIBLE_DEVICES=0 python none.py --arugment_you_want
CUDA_VISIBLE_DEVICES=0 python joint.py --arugment_you_want
```

and following previous works:

- **CLeWI**: Modified from [this GitHub repository](https://github.com/jedrzejkozal/weight-interpolation-cl), work from "[Continual Learning with Weight Interpolation](https://arxiv.org/abs/2404.04002)."
- **WSC**: Modified from [this GitHub repository](https://github.com/umamicode/weight-space-consolidation), work from "[Forget Forgetting: Continual Learning in a World of Abundant Memory](https://arxiv.org/abs/2502.07274)."
- **Generative Replay (GR)**: Modified from [this GitHub repository](https://github.com/msrocean/continual-learning-malware), work from "[]()."
- **EWC**: Modified from [this GitHub repository](https://github.com/msrocean/continual-learning-malware), work from "[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)."
- **LwF**: Modified from [this GitHub repository](https://github.com/msrocean/continual-learning-malware), work from "[Learning without Forgetting](https://arxiv.org/pdf/1606.09282)."
- **iCaRL**: Modified from [this GitHub repository](https://github.com/msrocean/continual-learning-malware), work from "[iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)."
- **TAMiL**: Modified from [this GitHub repository](https://github.com/NeurAI-Lab/TAMiL), work from "[Task-Aware Information Routing from Common Representation Space in Lifelong Learning](https://arxiv.org/abs/2302.11346)."
- **MalCL**: Modified from [this GitHub repository](https://github.com/MalwareReplayGAN/MalCL), work from "[MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification](https://arxiv.org/abs/2501.01110)."

You can run baselines with the following command:

```python
# CLeWI for EMBER-Class and AZ-Class
cd ./baselines/CLeWI
CUDA_VISIBLE_DEVICES=6 python main.py --model="clewi" \
  --dataset="seq_ember" --n_tasks=11 \
  --lr=0.001 --buffer_size=500 --n_epochs=50 \
  --seed=42 --optim_wd=0.0 --optim_mom=0.0 \
  --batch_size=512 --sub_dataset="ember" # ember or az for --sub_dataset
```

```python
# WSC for EMBER-Class
cd ./baselines/WSC
MODEL_NAME=wsc_20_ember
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json
```

```python
# WSC for AZ-Class
cd ./baselines/WSC
MODEL_NAME=wsc_20_az
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json
```

```python
# GR for EMBER-Class
cd ./baselines/GR_EWC_LwF_EMBER
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=EMBER --tasks=11 --replay=generative --metrics --logger_file gr --scenario=class

# GR for AZ-Class
cd ./baselines/GR_EWC_LwF_AZ
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=ANDROZOO --tasks=11 --replay=generative --metrics --logger_file gr --scenario=class
```

```python
# EWC for EMBER-Class
cd ./baselines/GR_EWC_LwF_EMBER
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=EMBER --tasks=11 --ewc --lambda=50 --metrics --logger_file ewc --scenario=class

# EWC for AZ-Class
cd ./baselines/GR_EWC_LwF_AZ
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=ANDROZOO --tasks=11 --ewc --lambda=50 --metrics --logger_file ewc --scenario=class
```

```python
# LwF for EMBER-Class
cd ./baselines/GR_EWC_LwF_EMBER
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=EMBER  --tasks=11 --replay=current --distill --metrics --logger_file lwf --scenario=class

# LwF for AZ-Class
cd ./baselines/GR_EWC_LwF_AZ
CUDA_VISIBLE_DEVICES=0 python main.py --data_set=AZ  --tasks=11 --replay=current --distill --metrics --logger_file lwf --scenario=class
```

```python
# iCaRL
cd ./baselines/iCaRL

```

```python
# TAMiL
cd ./baselines/TAMiL

```

```python
# MalCL
cd ./baselines/MalCL

```