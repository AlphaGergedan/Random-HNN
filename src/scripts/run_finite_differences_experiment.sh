#!/bin/bash

echo "This is for solving PDE with finite-differences data (trajectory data of q and p)"
echo "The simulation will run using symplectic euler scheme as training integrator,"
echo "and use dt=1e-4 for simulating the true flow, and h=0.05,0.1,0.2,0.4,0.8 as observation timestep."
echo "WARNING: read the script before running, it starts background processes and may take a long time to finish."
echo "Select integrator to simulate the true flow: 'rk45' or 'symplectic_euler'"
read trueflowintegrator
echo "select output path to save the results:"
read output_path

# dt_observations=0.05, dt_flow_true=1e-4, without post-training correction
python src/main.py 'single_pendulum' -usefd -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.05 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.05, dt_flow_true=1e-4, with post-training correction
python src/main.py 'single_pendulum' -usefd -correct -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.05 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.1, dt_flow_true=1e-4, without post-training correction
python src/main.py 'single_pendulum' -usefd -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.1 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.1, dt_flow_true=1e-4, with post-training correction
python src/main.py 'single_pendulum' -usefd -correct -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.1 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.2, dt_flow_true=1e-4, without post-training correction
python src/main.py 'single_pendulum' -usefd -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.2 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.2, dt_flow_true=1e-4, with post-training correction
python src/main.py 'single_pendulum' -usefd -correct -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.2 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.4, dt_flow_true=1e-4, without post-training correction
python src/main.py 'single_pendulum' -usefd -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.4 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.4, dt_flow_true=1e-4, with post-training correction
python src/main.py 'single_pendulum' -usefd -correct -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.4 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.8, dt_flow_true=1e-4, without post-training correction
python src/main.py 'single_pendulum' -usefd -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.8 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &

# dt_observations=0.8, dt_flow_true=1e-4, with post-training correction
python src/main.py 'single_pendulum' -usefd -correct -trainintegrator='symplectic_euler' -trueflowintegrator=$trueflowintegrator \
                                     -timestepobservations=0.8 -timestepflowtrue=1e-4 -dof 1 -trainsetsize 2500 \
                                     -qtrainlimstart -3.14 -qtrainlimend 3.14 -ptrainlimstart -1 -ptrainlimend 1 \
                                     -testsetsize 10000 -qtestlimstart -3.14 -qtestlimend 3.14 -ptestlimstart -1 -ptestlimend 1 \
                                     -repeat 10 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' \
                                     -rcond=1e-13 -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472  \
                                     $output_path &
