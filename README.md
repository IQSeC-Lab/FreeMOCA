<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis </h1>

<p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> [<a href="https://arxiv.org/abs/2605.09664" target="_blank">ArXiv Paper Link</a>] </p>

<p align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> <b> Zahra Asadi*, Haeseung Jeon*, Sohyun Han, Md Mahmuduzzaman Kamol, Se Eun Oh, Mohammad Saidur Rahman† </b> </p>

<p align='center' style="text-align:center; font-size:2.0em;letter-spacing:2.0px;"> *Equally credited authors.  †Corresponding author. </p>


> [!NOTE]
> This is official implementation of the paper *FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis*.


## FreeMOCA Pipeline



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
# EMBER
FreeMOCA/data/EMBER
├── XY_train.npz
└── XY_test.npz

# Androzoo (AZ)
FreeMOCA/data/AZ
├── AZ_Class_Train.npz
└── AZ_Class_Test.npz
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
cd ./FreeMOCA-Class/EMBER-Class
CUDA_VISIBLE_DEVICES=0 python main.py --train_data /path/to/data --test_data /path/to/data
```

```python
# AZ Class-IL
cd ./FreeMOCA-Class/AZ-Class
CUDA_VISIBLE_DEVICES=0 python main.py --train_data /path/to/data --test_data /path/to/data
```

```python
# EMBER Domain-IL
cd ./FreeMOCA-Domain/EMBER-Domain
CUDA_VISIBLE_DEVICES=0 python main.py --data_root /path/to/data/directory
```

```python
# AZ Domain-IL
cd ./FreeMOCA-Domain/AZ-Domain
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

and following previous works:

- **iCaRL**
- **Experience Replay (ER)**
- **Generative Replay (GR)**
- **BI-R**
- **TAMiL**
- **MaLCL** 