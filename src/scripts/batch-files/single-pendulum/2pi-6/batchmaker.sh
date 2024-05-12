#!/bin/bash

# batchmaker scripts for generating batch scripts

q_train=200
p_train=200

while [ $q_train -ge 10 ]
do
	num_points=$((q_train * p_train))
	filename="solvehamil_"$num_points"_single_pendulum_2pi_6.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_"$num_points"_single_pendulum_2pi_6.sh
#SBATCH -o "$SCRATCH"/solve-hamil/single-pendulum/2pi-6/"$num_points"_single_pendulum.txt
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
#SBATCH --time=20:30:00

module load slurm_setup

echo \$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py -includebias 'single_pendulum' -dof 1 -qtrain $q_train -ptrain $p_train -qtrainlimstart -6.28 -qtrainlimend 6.28 -ptrainlimstart -6 -ptrainlimend 6 -qtest 200 -ptest 200 -qtestlimstart -6.28 -qtestlimend 6.28 -ptestlimstart -6 -ptestlimend 6 -repeat 100 -nneurons 1500 -nhiddenlayers 1 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-6.28 -elmbiasend=6.28 -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/single-pendulum/2pi-6/">>$filename

	# (optional, use with care) submit the batch script
	sbatch $filename
	q_train=$((q_train - 10))
	p_train=$((p_train - 10))
done
