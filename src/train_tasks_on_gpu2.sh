#!bash

export CUDA_VISIBLE_DEVICES=2


# Ablation study on lambda
for lambda in 0.0 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train.py train --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --lambda ${lambda} \
        --suffix "p2v_lambda${lambda}" \
        --tb_on --tb_intvl 10000000 <<< ""
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --lambda ${lambda} \
        --resume "../exp/svcd/weights/model_best_p2v_lambda${lambda}.pth" \
        --subset test --save_on --out_dir p2v_lambda${lambda/./''}
done