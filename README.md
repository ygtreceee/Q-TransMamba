# Q-TransMamba: Query-linked Hybrid Transformer-Mamba Model for Point Cloud based 3D Object Detection
	


## Usage
### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation.

### Dataset Preparation
Please follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). We adopt the same data generation process.

### Training
```
# multi-gpu training
cd tools
./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_M_124_fp32.yaml [other optional arguments]
```

### Testing
```
# multi-gpu testing
cd tools
bash scripts/dist_test.sh 8 --cfg_file <CONFIG_FILE> --ckpt <CHECKPOINT_FILE>
```
