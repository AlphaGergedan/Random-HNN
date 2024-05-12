#!/bin/bash

# batchmaker scripts for generating batch scripts

train_set_size=10000

while [ $train_set_size -ge 500 ]
do
	filename="solvehamil_"$train_set_size"_single_pendulum_2pi_1_n1500.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_"$train_set_size"_single_pendulum_2pi_1_n1500.sh
#SBATCH -o "$SCRATCH"/solve-hamil/single-pendulum/2pi-1/n1500/"$train_set_size"_single_pendulum.txt
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

echo \$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'single_pendulum' -dof 1 -trainsetsize $train_set_size -qtrainlimstart -6.28 -qtrainlimend 6.28 -ptrainlimstart -1 -ptrainlimend 1 -testsetsize 10000 -qtestlimstart -6.28 -qtestlimend 6.28 -ptestlimstart -1 -ptestlimend 1 -repeat 100 -includebias -nhiddenlayers 1 -nneurons 1500 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-6.28 -elmbiasend=6.28 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/single-pendulum/2pi-1/n1500">>$filename

	# (optional, use with care) submit the batch script
	sbatch $filename
	train_set_size=$((train_set_size - 500))
done
