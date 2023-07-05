#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

#echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#
#echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"

if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

dataset=NCI1

# GPSwLPE
for batch_size in 128; do
  for hidden in 64 128; do
    for head in 4 8; do
      CUDA_VISIBLE_DEVICES=$cuda python main.py --cfg scripts/enzymes-GPSwLPE.yaml dataset.name $dataset train.batch_size $batch_size gt.dim_hidden $hidden gnn.dim_inner $hidden gt.n_heads $head wandb.use False
    done
  done
done

# Graphormer
for batch_size in 128; do
  for hidden in 80 128; do
    for head in 8; do
      CUDA_VISIBLE_DEVICES=$cuda python main.py --cfg scripts/enzymes-Graphormer.yaml dataset.name $dataset train.batch_size $batch_size gt.dim_hidden $hidden gnn.dim_inner $hidden gt.n_heads $head wandb.use False
    done
  done
done

# SAN
for batch_size in 128; do
  for hidden in 64 128; do
    for head in 4 8; do
      CUDA_VISIBLE_DEVICES=$cuda python main.py --cfg scripts/enzymes-SAN.yaml dataset.name $dataset train.batch_size $batch_size gt.dim_hidden $hidden gnn.dim_inner $hidden gt.n_heads $head wandb.use False
    done
  done
done
