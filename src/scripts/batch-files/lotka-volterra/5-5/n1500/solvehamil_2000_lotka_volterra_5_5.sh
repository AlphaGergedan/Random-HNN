#!/bin/bash
#SBATCH -J solvehamil_2000_lotka_volterra_5_5.sh
#SBATCH -o /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/5-5/n1500/2000_lotka_volterra.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --partition=mpp3_batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
# 256 is the maximum reasonable value for CooLMUC-3
#SBATCH --mail-type=end
#SBATCH --mail-user=rahma@in.tum.de
#SBATCH --export=NONE
#SBATCH --time=05:00:00

module load slurm_setup

echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'lotka_volterra' -dof 1 -trainsetsize 2000 -qtrainlimstart -5 -qtrainlimend 5 -ptrainlimstart -5 -ptrainlimend 5 -testsetsize 10000 -qtestlimstart -5  -qtestlimend 5 -ptestlimstart -5 -ptestlimend 5 -repeat 100 -includebias -nhiddenlayers 1 -nneurons 1500 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-5 -elmbiasend=5 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/5-5/n1500
