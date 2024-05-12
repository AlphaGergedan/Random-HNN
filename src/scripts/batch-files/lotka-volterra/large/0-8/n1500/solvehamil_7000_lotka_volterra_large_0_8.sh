#!/bin/bash
#SBATCH -J solvehamil_7000_lotka_volterra_large_0_8.sh
#SBATCH -o /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/large/0-8/n1500/7000_lotka_volterra_large.txt
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

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'lotka_volterra_large' -dof 1 -trainsetsize 7000 -qtrainlimstart 0 -qtrainlimend 8 -ptrainlimstart 0 -ptrainlimend 8 -testsetsize 10000 -qtestlimstart 0  -qtestlimend 8 -ptestlimstart 0 -ptestlimend 8 -repeat 100 -includebias -nhiddenlayers 1 -nneurons 1500 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=0 -elmbiasend=8 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/large/0-8/n1500
