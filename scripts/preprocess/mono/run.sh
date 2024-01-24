#!/bin/bash
{
    REPO_ROOT="$1"
    DATA_ROOT="$2"
    SAVE_ROOT="$3"
    SCENE_ID="$4"
    MIDAS_TYPE="$5"

    printf '\nREPO_ROOT: %s' ${REPO_ROOT}
    printf '\nDATA_ROOT: %s' ${DATA_ROOT}
    printf '\nSAVE_ROOT: %s' ${SAVE_ROOT}
    printf '\nSCENE_ID: %s' ${SCENE_ID}
    printf '\nMIDAS_TYPE: %s' ${MIDAS_TYPE}

    eval "$(conda shell.bash hook)"
    conda activate pgdvs

    # pip install timm==0.6.12

    cd ${REPO_ROOT}
    export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

    ulimit -n 65000;
    ulimit -c 0;   # Disable core file creation

    export MKL_THREADING_LAYER=GNU;
    export NCCL_P2P_DISABLE=1;
    export HYDRA_FULL_ERROR=1;
    export OC_CAUSE=1;

    if [ ! -f ${REPO_ROOT}/checkpoints/midas/dpt_beit_large_512.pt ]; then
        wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt -P ${REPO_ROOT}/checkpoints/midas/
        wget https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt -P ${REPO_ROOT}/checkpoints/midas/
    fi

    if [ ! -f ${REPO_ROOT}/third_party/RAFT/models/raft-sintel.pth ]; then
        wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip -P ${REPO_ROOT}/third_party/RAFT/
        unzip ${REPO_ROOT}/third_party/RAFT/models.zip -d ${REPO_ROOT}/third_party/RAFT/
    fi

    python ${REPO_ROOT}/scripts/preprocess/mono/generate_frame_midas.py \
        --data_root ${DATA_ROOT} \
        --save_dir ${SAVE_ROOT} \
        --midas_ckpt_dir ${REPO_ROOT}/checkpoints/midas \
        --midas_model_type ${MIDAS_TYPE}
    
    python ${REPO_ROOT}/scripts/preprocess/mono/generate_flows.py \
        --data_root ${SAVE_ROOT} \
        --save_dir ${SAVE_ROOT}
    
    python ${REPO_ROOT}/scripts/preprocess/mono/generate_sequence_midas.py \
        --data_root ${SAVE_ROOT} \
        --save_dir ${SAVE_ROOT}

    exit;
}