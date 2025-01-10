# Boundary-Aware Task Prediction for Class-Incremental Learning

## Requirements

1. Create a Conda virtual environment named `BTP`:

```
conda create --name BTP python=3.8
```

2. Activate the Conda environment:

```
conda activate BTP
```

3. To install the necessary dependencies, run the following command:

```
pip install -r requirements.txt
```


## Training

This section provides instructions on how to train the BTP model using our code.

### Data

Before training and evaluation, please download the datasets (CIFAR-10, CIFAR-100, and TinyImageNet). By default, the working directory is set to `~/data` in our code. You can modify this path as needed.

### Pre-trained Model

We use the pre-trained DeiT model provided by [MORE](https://github.com/k-gyuhak/MORE). Download the model and save it as `./ckpt/pretrained/deit_small_patch16_224_in661.pth`. If you'd like to experiment with other pre-trained visual encoders, download them to the same directory (you can find the pre-trained weights in timm or Hugging Face). We also support pre-trained models for Dino, MAE, CILP, ViT (small, tiny), and DeiT (small, tiny).

### Training Scripts

There are five provided training and evaluation scripts, each designed for specific scenarios:

- `deit_small_in661_C10_5T.sh`: For training with CIFAR-10 and 5 tasks.
- `deit_small_in661_C100_10T.sh`: For training with CIFAR-100 and 10 tasks.
- `deit_small_in661_C100_20T.sh`: For training with CIFAR-100 and 20 tasks.
- `deit_small_in661_T_5.sh`: For a custom task setup with 5 tasks.
- `deit_small_in661_T_10.sh`: For a custom task setup with 10 tasks.

To execute a script, run the following command (replace `<script_name>` with the appropriate script for your scenario):

```bash
bash scripts/<script_name>
```

Each script performs both training and testing. The default training will train BTP for 5 random seeds. During training, the results will be logged in the `ckpt` directory, and the initial training results (without BTP inference techniques) will be stored as `$HAT_{CIL}$`. After evaluation, the results will be updated.
For example, the results for the first run with seed=2023 will be saved in `./ckpt/seq0/seed2023/progressive_main_2023`.

**Note**: The results in the paper were obtained using Nvidia RTX3090 GPUs with CUDA 11.3. Using different hardware or software versions may lead to slight performance variations.
