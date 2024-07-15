#!/bin/bash
#SBATCH -t ---time---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=---n---

#SBATCH -p ---p---
#SBATCH --output=/home/ftijhuis/errout/%A_sub----subject---_ses----session---_slice----data_portion---.out
#SBATCH --error=/home/ftijhuis/errout/%A_sub----subject---_ses----session---_slice----data_portion---.err

#SBATCH --mail-type=END
#SBATCH --mail-user=---email---

module load 2023
source activate prffitting

cd /home/ftijhuis/software/dnfitting/scripts
python prf_fitting_norm_snellius.py ---subject--- 128 ---data_portion--- ---session---

