#! /bin/bash

set -x
CONFIG=$1
GPUS=$2
GPU_IDS=$3
MASTER_PORT=$(( (RANDOM % 48128) + 1024 ))
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} train.py --launcher pytorch  \
    --cfg_file $CONFIG \
    --batch_size 8 --workers 4 --epochs 20 \
    --ckpt_save_epoch_interval 1 --sync_bn \
    --wo_gpu_stat \
    --statement "QueryDeformableMambav5 on nus dataset(BALANCED_RESAMPLING=False). TransFusion with FP32. ['sum', 'sum', 'sum'] & z_res:[True, True, True] & skip:[True, False, True]. Directions: x-xxx-xxx-x. Num Query=1500. LR\DIV\PCT\CLIP=0.004\10\0.4\35(epochs=20). WinPosEmbed & GELU & BN & NO Dropout & TransLayers=1. M(1,2,4). Add ffn behind each QSSM & All ffn_rate=4 & SimpleFFN. topkr=0.12. 6008" \
    # --statement "QueryDeformableMambav5 on nus dataset(BALANCED_RESAMPLING=False). TransFusion with FP32. ['sum', 'sum', 'sum'] & z_res:[True, True, True] & skip:[True, False, True]. Directions: x-xxx-xxx-x. Num Query=1500. LR\DIV\PCT\CLIP=0.004\10\0.4\35(epochs=20). WinPosEmbed & GELU & BN & NO Dropout & TransLayers=1. M(1,2,4). Add ffn behind each QSSM & All ffn_rate=4 & SimpleFFN. topkr=0.10. 6007" \

    # --fp16 \
    # --ckpt /raid5/ygtrece/LION/output/train/qdefmamba_centerpoint_waymo_v5_M_20250710-061520/ckpt/latest_model.pth \
    # --autograd_detect_mode \
    # --vis_save_interval 1000 \

# Check:    
# 1. CUDA 11.8
# 2. Env: lion in 241
# ./dist_train_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_centerpoint_waymo_v5_M_fp32.yaml 2 3,4

# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_M_124_fp32.yaml 4 0,1,2,3
# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_M_124_fp32.yaml 4 4,5,6,7
# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_M_124_fp32.yaml 8 0,1,2,3,4,5,6,7

# ./dist_train_nus_fp32.sh ../output/train/qdefmamba_transfusion_nus_v5_M_124_fp32_20250730-123800/qdefmamba_transfusion_nus_v5_M_124_fp32.yaml 4 0,1,2,3

# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_S_124_fp32.yaml 4 0,1,2,3
# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_S_222_fp32.yaml 2 4,5
# ./dist_train_nus_fp32.sh ./cfgs/qdefmamba_models/qdefmamba_transfusion_nus_v5_S_224_fp32.yaml 2 6,7