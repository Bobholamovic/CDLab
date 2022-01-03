#!bash

set -e

cd ../src

# LEVIR-CD
python train.py train --exp_config ../configs/levircd/config_levircd_cdnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_ifn.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_snunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_stanet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_lunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/levircd/config_levircd_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# OSCD
python train.py train --exp_config ../configs/oscd/config_oscd_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/oscd/config_oscd_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/oscd/config_oscd_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# SVCD
python train.py train --exp_config ../configs/svcd/config_svcd_cdnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_ifn.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_snunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_stanet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_lunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_dsamnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/config_svcd_bit.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# AirChange-Szada
python train.py train --exp_config ../configs/szada/config_szada_cdnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/szada/config_szada_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/szada/config_szada_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/szada/config_szada_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# AirChange-Tiszadob
python train.py train --exp_config ../configs/tiszadob/config_tiszadob_cdnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/tiszadob/config_tiszadob_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/tiszadob/config_tiszadob_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/tiszadob/config_tiszadob_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# WHU
python train.py train --exp_config ../configs/whu/config_whu_cdnet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_siamunet-conc.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_siamunet-diff.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_unet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_ifn.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_snunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_stanet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_lunet.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/whu/config_whu_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on