#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu
#SBATCH --job-name=ectc
#SBATCH --mail-type=END
#SBATCH --mail-user=axs10302@nyu.edu
#SBATCH --time=03:15:00
#SBATCH --output=stdout.out

RUNDIR=$SCRATCH/ectc/run-${SLURM_JOB_ID/.*}
echo $RUNDIR

module purge 

singularity exec --nv \
	--overlay $SCRATCH/pytorch-example/my_pytorch.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	/bin/bash -c "source /ext3/env.sh; 
python pretrain_ectc.py"

#output if a cuda device is online, exit if no gpu is allocated: 

sleep 15 




