#!/bin/bash -l

# Slurm submit script
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

conda activate
conda activate viewpoint

srun python demoObjectron.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --object bottle --split train --confidence-threshold 0.3 --dataSave True --opts MODEL.WEIGHTS ../weights/model_final_280758.pkl
srun python demoObjectron.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --object bottle --split test --confidence-threshold 0.3 --dataSave True --opts MODEL.WEIGHTS ../weights/model_final_280758.pkl