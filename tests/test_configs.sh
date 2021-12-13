#!bash

set -e

cd ../src

# LEVIR-CD
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
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_2donly.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_alter.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_decouple.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_halfhalf.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_latefusion.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_notemporal.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v_perm.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on
python train.py train --exp_config ../configs/svcd/p2v/config_svcd_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

# SYSU-CD
python train.py train --exp_config ../configs/sysucd/config_sysucd_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on

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
python train.py train --exp_config ../configs/whu/config_whu_p2v.yaml --batch_size 1 --num_epochs 1 --num_workers 0 --repeats 1 --log_off --debug_on