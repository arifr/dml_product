#!/bin/bash
#
#SBATCH --job-name=dml
#SBATCH --output=dml-%j.out
#SBATCH --error=err.out



#srun python3 Standard_Training.py --gpu 0 --savename resnet_op_margin_dist     --dataset online_products --n_epochs 40 --tau 30    --loss marginloss --sampling distance
#srun python3 Standard_Training.py --gpu 0 --savename resnet_op_margin_dist     --dataset cub200 --n_epochs 20 --tau 30    --loss marginloss --sampling distance
#srun python3 Standard_Training.py --gpu 0 --savename resnet_op_margin_dist     --dataset cust-shop  --n_epochs 40 --tau 30    --loss marginloss --sampling distance
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner all



#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner all --embedding 64
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner all --embedding 128 
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner all --embedding 256 
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner all --embedding 512


#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner semihard --embedding 64
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner semihard --embedding 128
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner semihard --embedding 256
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner semihard --embedding 512



#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner easy_positif --embedding 64
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner easy_positif --embedding 128
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner easy_positif --embedding 256
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner easy_positif --embedding 512


#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 64
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 128
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 256
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 512

#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_class.py --miner semihard --embedding 256 --model-fname '/home/m405305/programs/custom_loss/log/train/m2_new_model_MobileNetV2_SOP_96_100_0.001.pth'

#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 64
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 128
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner batch_hard  --embedding 256
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner pair_margin --loss pair --embedding 512
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner pair_margin --loss pair --embedding 256
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner pair_margin --loss pair --embedding 128
#srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_main.py --miner pair_margin --loss pair --embedding 64

srun /home/m405305/miniconda3/bin/python /home/m405305/programs/dml_product/dml_product_trans.py --miner pair_margin --loss pair --embedding 512 --model-fname 


