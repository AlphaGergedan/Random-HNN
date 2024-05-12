#!/bin/bash

# batchmaker scripts for generating batch scripts

q_train=240
p_train=100

#q_train=228

while [ $q_train -ge 12 ]
do
	num_points=$((q_train * p_train))
	filename="solvehamil_"$num_points"_single_pendulum_f15_pi_0.5.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_"$num_points"_single_pendulum_f15_pi_0.5.sh
#SBATCH -o "$SCRATCH"/solve-hamil/single-pendulum/f15/pi-0.5/"$num_points"_single_pendulum.txt
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
#SBATCH --time=08:00:00

module load slurm_setup

echo \$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py -includebias 'single_pendulum_15_freq' -dof 1 -qtrain $q_train -ptrain $p_train -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -0.5 -ptrainlimend 0.5 -qtest 240 -ptest 100 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -0.5 -ptestlimend 0.5 -repeat 100 -nneurons 1500 -nhiddenlayers 1 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/single-pendulum/f15/pi-0.5/">>$filename

	# (optional, use with care) submit the batch script
	sbatch $filename
	q_train=$((q_train - 12))
	p_train=$((p_train - 5))
done
