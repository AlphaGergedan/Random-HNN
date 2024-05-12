#!/bin/bash

# batchmaker scripts for generating batch scripts

train_set_size=10000
n_nodes=2000

while [ $n_nodes -ge 100 ]
do
	filename="solvehamil_n"$n_nodes"_d"$train_set_size"_single_pendulum_f20_pi_0.5.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_n"$n_nodes"_d"$train_set_size"_single_pendulum_f20_pi_1.sh
#SBATCH -o "$SCRATCH"/solve-hamil/single-pendulum/f20/pi-0.5/d10000/"$n_nodes"_single_pendulum.txt
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

echo \$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'single_pendulum_20_freq' -dof 1 -trainsetsize $train_set_size -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -0.5 -ptrainlimend 0.5 -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -0.5 -ptestlimend 0.5 -repeat 100 -includebias -nhiddenlayers 1 -nneurons $n_nodes -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/single-pendulum/f20/pi-0.5/d10000">>$filename 

	# (optional, use with care) submit the batch script
	sbatch $filename
	n_nodes=$((n_nodes - 100))
done
