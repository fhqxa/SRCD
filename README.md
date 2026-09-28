# SRCD

[Graphical abstract-SRCD.pdf](https://github.com/user-attachments/files/32740125/Graphical.abstract-SRCD.pdf)


## Requirements

We recommend using Conda to create the environment.

```bash
conda env create -f environment.yml
conda activate scl
```

The main dependencies include:

* Python 3.12
* PyTorch 2.5.1
* torchvision 0.20.1
* CUDA 12.4
* NumPy
* scikit-learn
* Pillow

The complete environment configuration is provided in `environment.yml`.

## Datasets

The datasets used in this project can be downloaded from:

[Download Datasets](https://www.dropbox.com/scl/fo/gd4q67blo0ivmvw09syvt/AIatBE-PqY7y30_50mnGU1s?rlkey=xzio3ay56fh8gky1jbjtnrzu3&e=1&dl=0)

After downloading, place the datasets under the `data/` directory.

The project supports the following datasets:

* miniImageNet
* tieredImageNet
* CIFAR-FS
* FC100

The expected project structure is:

```text
SRCD/
├── data/
│   ├── miniImageNet/
│   ├── tiered-imagenet/
│   ├── CIFAR-FS/
│   └── fc100/
│
├── datasets/
│   ├── cifarfs.py
│   ├── cub.py
│   ├── fc100.py
│   ├── miniimagenet.py
│   ├── samplers.py
│   └── tiered_imagenet.py
│
├── resnet.py
├── train.py
├── test.py
├── util.py
├── environment.yml
└── README.md
```

The dataset arguments used in the code are:

| Dataset        | Argument  |
| -------------- | --------- |
| miniImageNet   | `mini`    |
| tieredImageNet | `tiered`  |
| CIFAR-FS       | `cifarfs` |
| FC100          | `fc100`   |

## Training

To train SRCD, run:

```bash
python train.py
```

We recommend explicitly specifying the dataset and output directory. For example:

### miniImageNet

```bash
python train.py --dataset mini --gpu 0 --save-path ./checkpoints/mini
```

### tieredImageNet

```bash
python train.py --dataset tiered --gpu 0 --save-path ./checkpoints/tiered
```

### CIFAR-FS

```bash
python train.py --dataset cifarfs --gpu 0 --save-path ./checkpoints/cifarfs
```

### FC100

```bash
python train.py --dataset fc100 --gpu 0 --save-path ./checkpoints/fc100
```

During training, the latest checkpoint is saved as:

```text
checkpoint.pth.tar
```

and the best-performing checkpoint is saved as:

```text
max-acc.pth.tar
```

under the directory specified by `--save-path`.

## Evaluation

To evaluate a trained model, run:

```bash
python test.py
```

The testing script loads the following checkpoint:

```text
<save-path>/max-acc.pth.tar
```

Therefore, `--save-path` should point to the directory containing the trained model.

For example, to evaluate a model trained on miniImageNet:

```bash
python test.py --dataset mini --gpu 0 --save-path ./checkpoints/mini
```

For tieredImageNet:

```bash
python test.py --dataset tiered --gpu 0 --save-path ./checkpoints/tiered
```

For CIFAR-FS:

```bash
python test.py --dataset cifarfs --gpu 0 --save-path ./checkpoints/cifarfs
```

For FC100:

```bash
python test.py --dataset fc100 --gpu 0 --save-path ./checkpoints/fc100
```

## Few-Shot Evaluation

The few-shot setting can be controlled using:

```text
--way
--shot
--query
```

For example, for **5-way 1-shot** evaluation:

```bash
python test.py \
    --dataset mini \
    --way 5 \
    --shot 1 \
    --query 15 \
    --gpu 0 \
    --save-path ./checkpoints/mini
```

For **5-way 5-shot** evaluation:

```bash
python test.py \
    --dataset mini \
    --way 5 \
    --shot 5 \
    --query 15 \
    --gpu 0 \
    --save-path ./checkpoints/mini
```
