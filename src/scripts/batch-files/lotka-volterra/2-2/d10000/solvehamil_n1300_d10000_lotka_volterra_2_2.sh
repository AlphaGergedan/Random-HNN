#!/bin/bash
#SBATCH -J solvehamil_n1300_d10000_lotka_volterra_2_2.sh
#SBATCH -o /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/2-2/d10000/1300_lotka_volterra.txt
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
#SBATCH --time=10:00:00

module load slurm_setup

echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'lotka_volterra' -dof 1 -trainsetsize 10000 -qtrainlimstart -2 -qtrainlimend 2 -ptrainlimstart -2 -ptrainlimend 2 -testsetsize 10000 -qtestlimstart -2 -qtestlimend 2 -ptestlimstart -2 -ptestlimend 2 -repeat 100 -includebias -nhiddenlayers 1 -nneurons 1300 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-2 -elmbiasend=2 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 /gpfs/scratch/pr63so/ge49rev3/solve-hamil/lotka-volterra/2-2/d10000
