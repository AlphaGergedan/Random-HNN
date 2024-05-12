# Project structure.

```
.
├── rahma_master_thesis.pdf # thesis pdf
├── plots                   # includes final plots used in the thesis
├── src                     # R-HNN implementation using swimnetworks module
│   ├── main.py             # main function for running and saving R-HNN experiments
│   ├── activations         # activation functions implementations
│   ├── error_functions     # error functions implementations
│   ├── hamiltonians        # Hamiltonian system implementations used
│   ├── integrators         # integrator implementations
│   ├── scripts             # scripts for creating plots from the saved experiments
│   ├── utils               # util functions
│   └── swimnetworks        # submodule used, see [here](https://github.com/AlphaGergedan/swimnetworks)
├── notebooks               # includes notebooks used for integration plots
│   └── integration
└── environments.yml
```

# Setup

To create the required conda environment run: `conda env create --file=environments.yml`.

# Run and Save Experiments

R-HNN implementation (Alg. 2) can be found in `src/utils/swim.py` inside `hswim` function. It is like a wrapper around the swimnetworks module (see original repo [here](https://gitlab.com/felix.dietrich/swimnetworks)) to be able to act like an HNN. For HNN reference see the papers by [Greydanus et al.](https://proceedings.neurips.cc/paper/2019/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf), [Bertalan et al.](https://pubs.aip.org/aip/cha/article/29/12/121107/1027304).

For creating the scaling plots cluster provided by [lrz](www.lrz.de) is used. The batch files used to submit jobs can be found under `src/scripts/batch-files`.

The following examples are taken from the batch-files, note that the path to the `main.py` should be changed to the location of the `main.py` file, if you are in the project root of this repo and want to locally run, you should change it to `src/main.py`.

## Single Pendulum

```sh
python src/main.py 'single_pendulum' -dof 1 -trainsetsize 1000 -qtrainlimstart -6.28 -qtrainlimend 6.28 -ptrainlimstart -6 -ptrainlimend 6 \
                                            -testsetsize 5000 -qtestlimstart -6.28 -qtestlimend 6.28 -ptestlimstart -6 -ptestlimend 6 \
                                     -repeat 100 -includebias -nhiddenlayers 1 -nneurons 200 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 \
                                     -elmbiasstart=-6.28 -elmbiasend=6.28 -resampleduplicates \
                                     -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 \
                                     output_dir
```

## Lotka-Volterra

```sh
python src/main.py 'lotka_volterra' -dof 1 -trainsetsize 4000 -qtrainlimstart -5 -qtrainlimend 5 -ptrainlimstart -5 -ptrainlimend 5 \
                                           -testsetsize 10000 -qtestlimstart -5  -qtestlimend 5 -ptestlimstart -5 -ptestlimend 5 \
                                    -repeat 100 -includebias -nhiddenlayers 1 -nneurons 1500 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 \
                                    -elmbiasstart=-5 -elmbiasend=5 -resampleduplicates \
                                    -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 \
                                    output_dir
```

## Double Pendulum

```sh
python src/main.py 'double_pendulum' -dof 2 -trainsetsize 20000 -qtrainlimstart -3.14 -3.14 -qtrainlimend 3.14 3.14 -ptrainlimstart -1 -1 -ptrainlimend 1 1 \
                                            -testsetsize 20000 -qtestlimstart -3.14 -3.14 -qtestlimend 3.14 3.14 -ptestlimstart -1 -1 -ptestlimend 1 1 \
                                            -repeat 10 -includebias -nhiddenlayers 1 -nneurons 1000 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 \
                                            -elmbiasstart=-3.14 -elmbiasend=3.14 -resampleduplicates \
                                            -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 \
                                            output_dir
```

## Henon-Heiles

```sh
python src/main.py 'henon_heiles_0.7_alpha' -dof 2 -trainsetsize 20000 -qtrainlimstart -1 -1 -qtrainlimend 1 1 -ptrainlimstart -1 -1 -ptrainlimend 1 1 \
                                                   -testsetsize 20000 -qtestlimstart -1 -1 -qtestlimend 1 1 -ptestlimstart -1 -1 -ptestlimend 1 1 \
                                                   -repeat 10 -includebias -nhiddenlayers 1 -nneurons 3000 -activation 'tanh' -parametersampler 'tanh' -rcond=1e-13 \
                                                   -elmbiasstart=-1 -elmbiasend=1 -resampleduplicates \
                                                   -trainrandomseedstart=3943 -testrandomseedstart=29548 -modelrandomseedstart=992472 \
                                                   output_dir
```

# Figures

- All the scaling plots in the thesis can be created using the related script inside `src/scripts/batch-files`.
- For the integration plots (Fig. 3.11, Fig. 3.12, Fig. 3.14, Appendix Fig. 15-18, Appendix Fig. 20-23) for double pendulum and Henon-Heiles please refer to the notebooks under `notebooks/integration`. In order to create them locally you first need to run the experiment with the proper parameters. For the parameter values, please refer to the thesis.
- Fig. 3.15 can be created using the script `src/scripts/run_finite_differences_experiment.sh`, make sure to set the integrator to simulate the true flow to `rk45` for the same plot.

Tip: Setting `repeat` too large can lead to longer runtimes, better reduce it to test locally. Also, make sure you set `OMP_NUM_THREADS` to the number of CPU cores of your node, in order utilize full capacity for numpy operations.

For any questions do not hesitate to contact me at: rahma@in.tum.de.
