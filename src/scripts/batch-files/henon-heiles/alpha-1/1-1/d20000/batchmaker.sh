#!/bin/bash

# batchmaker scripts for generating batch scripts

train_set_size=20000
n_nodes=10000

while [ $n_nodes -ge 500 ]
do
	filename="solvehamil_n"$n_nodes"_d"$train_set_size"_henon_heiles_alpha_1_1_1.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_n"$n_nodes"_d"$train_set_size"_henon_heiles_alpha_1_1_1.sh
#SBATCH -o "$SCRATCH"/solve-hamil/henon-heiles/r10/alpha-1/1-1/d20000/"$n_nodes"_henon_heiles_alpha_1.txt
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

echo \$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'henon_heiles_1_alpha' -dof 2 -trainsetsize $train_set_size -qtrainlimstart -1 -1 -qtrainlimend 1 1 -ptrainlimstart -1 -1 -ptrainlimend 1 1 -testsetsize 20000 -qtestlimstart -1 -1 -qtestlimend 1 1 -ptestlimstart -1 -1 -ptestlimend 1 1 -repeat 10 -includebias -nhiddenlayers 1 -nneurons $n_nodes -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=-1 -elmbiasend=1 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/henon-heiles/r10/alpha-1/1-1/d20000">>$filename

	# (optional, use with care) submit the batch script
	sbatch $filename
	n_nodes=$((n_nodes - 500))
done
