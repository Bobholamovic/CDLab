#!bash

export CUDA_VISIBLE_DEVICES=4


# Ablation study on video_len
for video_len in 2 4 6 12 16
do
    python train.py train --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --p2v_model.video_len ${video_len} \
        --suffix "p2v_len${video_len}" \
        --tb_on --tb_intvl 10000000 <<< ""
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --p2v_model.video_len ${video_len} \
        --resume "../exp/svcd/weights/model_best_p2v_len${video_len}.pth" \
        --subset test --save_on --out_dir p2v_len${video_len}
done

# Ablation study on architecture
for arch in 2donly notemporal latefusion decouple
do
    python train.py train --exp_config "../configs/svcd/p2v/config_svcd_p2v_${arch}.yaml" \
        --tb_on --tb_intvl 10000000 <<< ""
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v_${arch}.yaml" \
        --resume "../exp/svcd/weights/model_best_p2v_${arch}.pth" \
        --subset test --save_on --out_dir p2v_${arch}
done