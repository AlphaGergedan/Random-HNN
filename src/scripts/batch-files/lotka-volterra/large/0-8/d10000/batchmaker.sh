#!/bin/bash

# batchmaker scripts for generating batch scripts

train_set_size=10000
n_nodes=2000

while [ $n_nodes -ge 100 ]
do
	filename="solvehamil_n"$n_nodes"_d"$train_set_size"_lotka_volterra_large_0_8.sh"

	echo $filename "created"

	echo "#!/bin/bash
#SBATCH -J solvehamil_n"$n_nodes"_d"$train_set_size"_lotka_volterra_large_0_8.sh
#SBATCH -o "$SCRATCH"/solve-hamil/lotka-volterra/large/0-8/d10000/"$n_nodes"_lotka_volterra.txt
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

python /dss/dsshome1/0B/ge49rev3/master-thesis/main.py 'lotka_volterra_large' -dof 1 -trainsetsize $train_set_size -qtrainlimstart 0 -qtrainlimend 8 -ptrainlimstart 0 -ptrainlimend 8 -testsetsize 10000 -qtestlimstart 0 -qtestlimend 8 -ptestlimstart 0 -ptestlimend 8 -repeat 100 -includebias -nhiddenlayers 1 -nneurons $n_nodes -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 -elmbiasstart=0 -elmbiasend=8 -resampleduplicates -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 $SCRATCH/solve-hamil/lotka-volterra/large/0-8/d10000">>$filename 

	# (optional, use with care) submit the batch script
	sbatch $filename
	n_nodes=$((n_nodes - 100))
done
