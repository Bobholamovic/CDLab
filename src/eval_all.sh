#!bash

export CUDA_VISIBLE_DEVICES=2

MODELS=(cdnet unet siamunet-conc siamunet-diff stanet lunet bit ifn snunet p2v)
DATASETS=(svcd levircd whu)


# Comparative experiments
for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        if [ ${dataset} = 'svcd' ] && [ ${model} = 'p2v' ]
        then
            cfg_path='p2v/config_svcd_p2v.yaml' 
        else
            cfg_path="config_${dataset}_${model}.yaml"
        fi
        if [ ${dataset} = 'levircd' ]
        then
            if [ ${model} = 'p2v' ]
            then
                python sw_test.py --exp_config "../configs/${dataset}/${cfg_path}" \
                    --ckp_path "../exp/${dataset}/weights/model_best_${model}.pth" \
                    --t1_dir "/home/gdf/data/LEVIR-CD/ori/test/A" \
                    --t2_dir "/home/gdf/data/LEVIR-CD/ori/test/B" \
                    --gt_dir "/home/gdf/data/LEVIR-CD/ori/test/label" \
                    --out_dir "../exp/levircd/sw_out/${model}" \
                    --save_on \
                    --stride 64 \
                    --mu 110.2008 100.63983 95.99475 \
                    --sigma 58.14765 56.46975 55.332195
            else
                python sw_test.py --exp_config "../configs/${dataset}/${cfg_path}" \
                    --ckp_path "../exp/${dataset}/weights/model_best_${model}.pth" \
                    --t1_dir "/home/gdf/data/LEVIR-CD/ori/test/A" \
                    --t2_dir "/home/gdf/data/LEVIR-CD/ori/test/B" \
                    --gt_dir "/home/gdf/data/LEVIR-CD/ori/test/label" \
                    --out_dir "../exp/levircd/sw_out/${model}" \
                    --save_on \
                    --stride 64 \
                    --mu 0.0 0.0 0.0 \
                    --sigma 255.0 255.0 255.0
            fi
        fi
        python train.py eval --exp_config "../configs/${dataset}/${cfg_path}" \
            --resume "../exp/${dataset}/weights/model_best_${model}.pth" \
            --subset test --save_on
    done
done


# Ablation studies
# on lambda
for lambda in 0.0 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --lambda ${lambda} \
        --resume "../exp/svcd/weights/model_best_p2v_lambda${lambda}.pth" \
        --subset test --save_on --out_dir p2v_lambda${lambda/./''}
done

# on video_len
for video_len in 2 4 6 12 16
do
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v.yaml" \
        --p2v_model.video_len ${video_len} \
        --resume "../exp/svcd/weights/model_best_p2v_len${video_len}.pth" \
        --subset test --save_on --out_dir p2v_len${video_len}
done

# on architecture
for arch in 2donly notemporal latefusion decouple
do
    python train.py eval --exp_config "../configs/svcd/p2v/config_svcd_p2v_${arch}.yaml" \
        --resume "../exp/svcd/weights/model_best_p2v_${arch}.pth" \
        --subset test --save_on --out_dir p2v_${arch}
done