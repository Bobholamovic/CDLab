#!bash

export CUDA_VISIBLE_DEVICES=3

MODELS=(cdnet unet siamunet-conc siamunet-diff stanet lunet bit ifn snunet p2v escnet)


# Comparative experiments
for model in "${MODELS[@]}"
do
    if [ ${model} = 'p2v' ]
    then
        cfg_path='p2v/config_svcd_p2v.yaml' 
    else
        cfg_path="config_svcd_${model}.yaml"
    fi
    python train.py train --exp_config "../configs/svcd/${cfg_path}" \
        --tb_on --tb_intvl 10000000 <<< ""
    python train.py eval --exp_config "../configs/svcd/${cfg_path}" \
        --resume "../exp/svcd/weights/model_best_${model}.pth" \
        --subset test --save_on
done