#!/bin/bash
#SBATCH -J solvehamil_n6500_d20000_double_pendulum_pi_1.sh
#SBATCH -o /gpfs/scratch/pr63so/ge49rev3/solve-hamil/double-pendulum/pi-1/r10/d20000/6500_double_pendulum.txt
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
#SBATCH --time=45:00:00

module load slurm_setup

echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'double_pendulum' -dof 2 -trainsetsize 20000 -qtrainlimstart -3.14 -3.14 -qtrainlimend 3.14 3.14 -ptrainlimstart -1 -1 -ptrainlimend 1 1 -testsetsize 20000 -qtestlimstart -3.14 -3.14 -qtestlimend 3.14 3.14 -ptestlimstart -1 -1 -ptestlimend 1 1 -repeat 10 -includebias -nhiddenlayers 1 -nneurons 6500 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 /gpfs/scratch/pr63so/ge49rev3/solve-hamil/double-pendulum/pi-1/r10/d20000
