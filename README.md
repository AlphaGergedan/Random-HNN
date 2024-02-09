Project structure.

```
.
├── books
├── build
├── code
│   ├── GP
│   ├── ELM
│   ├── notebooks
│   │   ├── nonlinear_pendulum
│   │   └── toy_example
│   └── swimnetworks
├── ode_examples
├── papers
├── PINNs
└── tex
```

# Linux Cluster

```sh
ssh -Y lxlogin1.lrz.de -l xxyyyzz

Haswell (CoolMUC-2) login node
ssh -Y lxlogin2.lrz.de -l xxyyyzz

Haswell (CoolMUC-2) login node
ssh -Y lxlogin3.lrz.de -l xxyyyzz

Haswell (CoolMUC-2) login node
ssh -Y lxlogin4.lrz.de -l xxyyyzz

Haswell (CoolMUC-2) login node
ssh -Y lxlogin8.lrz.de -l xxyyyzz	KNL Segment (CooMUC-3) login node
```

The login nodes are meant for preparing your jobs, developing your programs, and as a gateway for copying data from your own computer to the cluster and back again. Since this resource is shared among many users, LRZ requires that you do not start any long-running or memory-hogging programs on these nodes; production runs should use batch jobs (serial or parallel) that are submitted to the SLURM scheduler. Our SLURM configuration also supports semi-interactive testing. Violation of the usage restrictions on the login nodes may lead to your account being blocked from further access to the cluster, apart from your processes being forcibly removed by LRZ administrative staff!

Notes:
- The -Y option of ssh is responsible for tunneling of the X11 protocol, it may be omitted if no X11 clients are required, or if you already have otherwise configured X11 tunnelling in your ssh client.
- The HOME directory on the Linux Cluster is an NFS mounted volume, which is uniformly mounted on all cluster nodes.
- The login node lxlogin8.lrz.de for the KNL cluster is itself not a KNL system; you can develop and compile your software there, but if you optimized for KNL, you may not be able to execute the program on the login node itself, but must use an interactive or scripted SLURM job.

cm2_tiny -> cm2_tiny (56 gb per node)
inter -> cm2_inter_large_mem (for more memory up to 120gb per node)
inter -> cm4_inter_large_mem (for more memory up to 1000gb per node)
teramem (huge memory)

