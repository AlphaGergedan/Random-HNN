#!/bin/bash

# batchmaker scripts for generating batch scripts

q_train=120
p_train=50

# create from 1 to 56 threads
while [ $q_train -ge 12 ]
do
  fileName="batch_"$q_train"q_train"$p_train"p_train_single_pendulum.sh"

  echo $fileName "created"

  echo "#!/bin/bash
#SBATCH -J "$q_train"q_train_"$p_train"p_train_single_pendulum.sh
#SBATCH -o "$SCRATCH"/"$q_train"q_train_"$p_train"p_train_single_pendulum.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
# 56 is the maximum reasonable value for CooLMUC-2
#SBATCH --mail-type=end
#SBATCH --mail-user=rahma@in.tum.de
#SBATCH --export=NONE
#SBATCH --time=00:15:00

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate thesis

python /dss/dsshome1/0B/ge49rev3/master-thesis/main_2d.py python main.py 'single_pendulum' $q_train $p_train -6.28 6.28 -1 1 False 120 50 -6.28 6.28 -1 1 100 1500 'tanh' 'tanh' 1e-13 True">>$filename

  # (optional, use with care) submit the batch script
  #sbatch $fileName
  q_train=$((q_train - 12))
  p_train=$((p_train - 5))
done
