import argparse

# definitions in thesis


# definitions listed here
defs = {

    # GENERAL DEFINITIONS

    # NEURAL NETWORK
    "neural network": {
        "desc": "revolutionary machine learning model which consist of interconnected processing units (neurons) that work together to perform various tasks, such as pattern recognition, classification, regression and more.",
        "refs": "literally every paper",
    },
    # DEEP NEURAL NETWORK
    "deep network": {
        "desc": "Deep network with multiple layers between the input and output layers. Number of hidden layers #hidden is more than 1.",
        "refs": "literally every paper",
    },
    "deep neural network": {
        "desc": "Deep network with multiple layers between the input and output layers. Number of hidden layers #hidden is more than 1.",
        "refs": "literally every paper",
    },
    # SHALLOW NEURAL NETWORK
    "shallow network": {
        "desc": "Neural network with a single hidden layer #hidden equals 1. In practice, many modern neural networks are deep, as deep learning has achieved remarkable success in a wide range of applications. However, shallow networks can serve as a useful starting point for simpler tasks or as building blocks within more complex architectures.",
        "refs": "Sampling weights of deep neural networks",
    },
    "shallow neural network": {
        "desc": "Neural network with a single hidden layer #hidden equals 1. In practice, many modern neural networks are deep, as deep learning has achieved remarkable success in a wide range of applications. However, shallow networks can serve as a useful starting point for simpler tasks or as building blocks within more complex architectures.",
        "refs": "Sampling weights of deep neural networks",
    },
    # FULLY CONNECTED NEURAL NETWORK
    "fully connected neural network": {
        "desc": "A fully connected neural network, also known as a dense neural network or a feedforward neural network, is a type of artificial neural network (ANN) in which each neuron in one layer is connected to every neuron in the subsequent layer. In a fully connected neural network, information flows in one direction, from the input layer to one or more hidden layers and finally to the output layer, without loops or feedback connections. This makes fully connected networks a fundamental and straightforward architecture in neural network design.",
        "refs": "literally every paper",
    },
    "dense neural network": {
        "desc": "A fully connected neural network, also known as a dense neural network or a feedforward neural network, is a type of artificial neural network (ANN) in which each neuron in one layer is connected to every neuron in the subsequent layer. In a fully connected neural network, information flows in one direction, from the input layer to one or more hidden layers and finally to the output layer, without loops or feedback connections. This makes fully connected networks a fundamental and straightforward architecture in neural network design.",
        "refs": "literally every paper",
    },
    "feedforward neural network": {
        "desc": "A fully connected neural network, also known as a dense neural network or a feedforward neural network, is a type of artificial neural network (ANN) in which each neuron in one layer is connected to every neuron in the subsequent layer. In a fully connected neural network, information flows in one direction, from the input layer to one or more hidden layers and finally to the output layer, without loops or feedback connections. This makes fully connected networks a fundamental and straightforward architecture in neural network design.",
        "refs": "literally every paper",
    },
    # EXTREME LEARNING MACHINE
    "extreme learning machine": {
        "desc": " ELMs are particularly known for their simplicity and efficiency in training compared to traditional gradient-based methods for deep learning. They were introduced as a way to speed up the training of neural networks while maintaining competitive performance. They have a single hidden layer of neurons making them shallow neural networks. They have random weight initialization for the connections between the input layer and the hidden layer, these weights are often drawn from a uniform or Gaussian distribution. They only train output layer using traditional methods such as linear regression or least squares to find the optimal weights, this step is often computationally efficient. They are proven to be universal function approximators with adaptive growth in hidden layer. They are not as flexible as other NNs but they have fast training times.",
        "refs": "",
    },
    # BARRON FUNCTIONS
    "barron function": {
        "desc": "Barron functions, named after David Barron, are a class of mathematical functions used in the analysis of approximation errors and convergence rates of neural networks, particularly in the context of function approximation problems. These functions have been the subject of study in the field of machine learning and computational mathematics. Barron functions are known for their properties as universal approximators. This means that, under certain conditions, they can approximate a wide range of functions with arbitrary precision. Universal approximation is a desirable property for neural networks because it implies that the network can represent a diverse set of functions. Barron functions are Lipschitz-continuous. Barron functions grow at most linearly at infinity. For Barron functions that converges to 0 at infinity, they are also bounded. Every Barron function can be decomposed as the sum of a bounded and a positively one-homogenous function. Barron functions are universal approximators.",
        "refs": "Sampling weights of deep neural networks, Representation formulas and pointwise properties for barron functions",
    },
    # UNIVERSAL APPROXIMATOR
    "universal approximator": {
        "desc": "A universal approximator is a mathematical or computational model, often a function or a neural network, that has the capability to approximate any given continuous function with arbitrary accuracy, provided certain conditions are met.",
        "refs": "",
    },
    # INFIMUM
    "infimum": {
        "desc": "Greatest lower bound.",
        "refs": "literally every math paper related to convergence.",
    },
    # SUPREMUM
    "supremum": {
        "desc": "Least upper bound.",
        "refs": "literally every math paper related to convergence.",
    },
    # BARRON SPACE
    "barron space": {
        "desc": "Set of Barron functions.",
        "refs": "Sampling weights of deep neural networks.",
    },
    # ODE
    "ordinary differential equation": {
        "desc": "Equation represents change of one independent variable",
        "refs": "Sampling weights of deep neural networks.",
    },
    # PDE
    "partial differential equation": {
        "desc": "Equation represents change of multiple independent variables. We have an entire function changing over time.",
        "refs": "Sampling weights of deep neural networks.",
    },
    # GAUSSIAN PROCESS
    "gaussian process": {
        "desc": "A stochastic process of random variables with a Gaussian distribution. A stochastic process describes systems randomly changing over time. GP can be interpreted as a random distrbution over functions f(x) defined by a mean function m(x) and a postive definite covariance function k(x,x'). For comparison: multivariate Gaussian captures a finite number of jointly distributied Gaussians, GP doesn't have this limitation, its mean and covariance is defined by a function so it can change. Each input to GP is correlated with another input defined by the covariance function. Since functions have infinite input domain, GP can be interpreted as an infinite dimensional Gaussian random variable.",
        "refs": "Sampling weights of deep neural networks.",
    },
    # PHASE SPACE
    "phase space": {
        "desc": "is our system space. Complete set of possible system states in Hamiltonian systems, is symplectic.",
        "refs": "Sampling weights of deep neural networks.",
    },
    # PHASE FLOW
    "phase flow": {
        "desc": "how p and q of a system change as system evolves, preserves symplectic structure of the phase space.",
        "refs": "Sampling weights of deep neural networks.",
    },
    # PHASE map
    "phase map": {
        "desc": "a symplectic transformation that returns the system at a later time and preserves the symplectic structure of the phase space during transformation?",
        "refs": "Sampling weights of deep neural networks.",
    },






    # PAPERS: for paper abbreviations used under /papers
    "bertalan-2019": {
        "desc": "On learning Hamiltonian systems from data. Abstract: Concise, accurate descriptions of physical systems through their conserved quantities abound in the natural sciences. In data science, however, current research often focuses on regression problems, without routinely incorporating additional assumptions about the system that gener- ated the data. Here, we propose to explore a particular type of underlying structure in the data: Hamiltonian systems, where an “energy” is conserved. Given a collection of observations of such a Hamiltonian system over time, we extract phase space coordinates and a Hamiltonian function of them that acts as the generator of the system dynamics. The approach employs an autoencoder neural network component to estimate the transformation from observations to the phase space of a Hamiltonian system. An additional neural network component is used to approximate the Hamiltonian function on this constructed space, and the two components are trained jointly. As an alternative approach, we also demonstrate the use of Gaussian processes for the estimation of such a Hamiltonian. After two illustrative examples, we extract an underlying phase space as well as the generating Hamiltonian from a collection of movies of a pendulum. The approach is fully data-driven and does not assume a particular form of the Hamiltonian function.",
        "refs": "",
    },
    "bolager-2023": {
        "desc": "Sampling weights of deep neural networks. Abstract: We introduce a probability distribution, combined with an efficient sampling algo- rithm, for weights and biases of fully-connected neural networks. In a supervised learning context, no iterative optimization or gradient computations of internal network parameters are needed to obtain a trained network. The sampling is based on the idea of random feature models. However, instead of a data-agnostic distri- bution, e.g., a normal distribution, we use both the input and the output training data of the supervised learning problem to sample both shallow and deep networks.  We prove that the sampled networks we construct are universal approximators. We also show that our sampling scheme is invariant to rigid body transformations and scaling of the input data. This implies many popular pre-processing techniques are no longer required. For Barron functions, we show that the L2-approximation error of sampled shallow networks decreases with the square root of the number of neurons. In numerical experiments, we demonstrate that sampled networks achieve comparable accuracy as iteratively trained ones, but can be constructed orders of magnitude faster. Our test cases involve a classification benchmark from OpenML, sampling of neural operators to represent maps in function spaces, and transfer learning using well-known architectures.",
        "refs": "",
    },
    "chen-2020": {
        "desc": "Symplectic Recurrent Neural Networks (SRNNs). Abstract: We propose Symplectic Recurrent Neural Networks (SRNNs) as learning algorithms that capture the dynamics of physical systems from observed trajectories. An SRNN models the Hamiltonian function of the system by a neural network and furthermore leverages symplectic integration, multiple- step training and initial state optimization to address the challenging nu- merical issues associated with Hamiltonian systems. We show SRNNs suc- ceed reliably on complex and noisy Hamiltonian systems. We also show how to augment the SRNN integration scheme in order to handle stiff dynamical systems such as bouncing billiards.",
        "refs": "",
    },
    "greydanus-2019": {
        "desc": "Hamiltonian Neural Networks (HNNs). Abstract: Even though neural networks enjoy widespread use, they still struggle to learn the basic laws of physics. How might we endow them with better inductive biases? In this paper, we draw inspiration from Hamiltonian mechanics to train models that learn and respect exact conservation laws in an unsupervised manner. We evaluate our models on problems where conservation of energy is important, including the two-body problem and pixel observations of a pendulum. Our model trains faster and generalizes better than a regular neural network. An interesting side effect is that our model is perfectly reversible in time.",
        "refs": "",
    },
    "horn-2022": {
        "desc": "Structure-Preserving Neural Networks For the N-body Problem. Key words: Neural Networks, Structure-Preserving Computing, Symplectic Algorithms, As- trophysics Abstract. In order to understand when it is useful to build physics constraints into neural net- works, we investigate different neural network topologies to solve the N -body problem. Solving the chaotic N -body problem with high accuracy is a challenging task, requiring special numerical integrators that are able to approximate the trajectories with extreme precision. In [1] it is shown that a neural network can be a viable alternative, offering solutions many orders of magnitude faster. Specialized neural network topologies for applications in scientific computing are still rare compared to specialized neural networks for more classical machine learning applications.  However, the number of specialized neural networks for Hamiltonian systems has been growing significantly during the last years [3, 5]. We analyze the performance of SympNets introduced in [5], preserving the symplectic structure of the phase space flow map, for the prediction of trajectories in N -body systems. In particular, we compare the accuracy of SympNets against standard multilayer perceptrons, both inside and outside the range of training data. We analyze our findings using a novel view on the topology of SympNets. Additionally, we also compare SympNets against classical symplectic numerical integrators. While the benefits of symplectic integrators for Hamiltonian systems are well understood, this is not the case for SympNets.",
        "refs": "",
    },
    "jin-2020": {
        "desc": "SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems. Abstract: We propose new symplectic networks (SympNets) for identifying Hamiltonian systems from data based on a composition of linear, activation and gradient modules. In particular, we define two classes of SympNets: the LA-SympNets composed of linear and activation modules, and the G-SympNets composed of gradient modules. Correspondingly, we prove two new universal approximation theorems that demonstrate that SympNets can approximate arbitrary symplectic maps based on appropriate activation functions. We then perform several experiments including the pendulum, double pendulum and three-body problems to investigate the expressivity and the generalization ability of SympNets.  The simulation results show that even very small size SympNets can generalize well, and are able to handle both separable and non-separable Hamiltonian systems with data points resulting from short or long time steps. In all the test cases, SympNets outperform the baseline models, and are much faster in training and prediction. We also develop an extended version of SympNets to learn the dynamics from irregularly sampled data. This extended version of SympNets can be thought of as a universal model representing the solution to an arbitrary Hamiltonian system.",
        "refs": "",
    },
    "pang-2020": {
        "desc": "Physics-Informed Learning Machines for Partial Differential Equations: Gaussian Processes Versus Neural Networks. Abstract:  We review and compare physics-informed learning models built upon Gaussian processes (GP) and deep neural networks (NN) for solving forward and inverse problems governed by linear and nonlinear PDEs. We define a unified data model on which GP, physics-informed GP (PIGP), NN, and physics-informed NN (PINN) are based. We develop continuous-time and discrete-time models to facilitate different application scenarios. We present a connection between a GP and an infinitely wide NN, which enables us to obtain a “best” kernel, which is determined directly by the data. We demon- strate the implementation of PIGP and PINN using a pedagogical example.  Additionally, we compare PIGP and PINN for two nonlinear PDEs, i.e., the 1D Burgers’ equation and the 2D Navier-Stokes, and provide guidance in choosing the proper machine learning model according to the problem type, i.e., forward or inverse problem, and the availability of data. These new meth- ods for solving PDEs governing multi-physics problems do not require any grid, and they are simple to implement, and agnostic to specific application.  Hence, we expect that variants and proper extensions of these methods will find broad applicability in the near future across different scientific disciplines but also in industrial applications.",
        "refs": "",
    },
    "raissi-2019": {
        "desc": "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Abstract: We introduce physics-informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given laws of physics described by general nonlinear partial differential equations. In this work, we present our developments in the context of solving two main classes of problems: data-driven solution and data-driven discovery of partial differential equations. Depending on the nature and arrangement of the available data, we devise two distinct types of algorithms, namely continuous time and discrete time models. The first type of models forms a new family of data-efficient spatio-temporal function approximators, while the latter type allows the use of arbitrarily accurate implicit Runge–Kutta time stepping schemes with unlimited number of stages. The effectiveness of the proposed framework is demonstrated through a collection of classical problems in fluids, quantum mechanics, reaction–diffusion systems, and the propagation of nonlinear shallow-water waves.",
        "refs": "",
    },
    "sazulibarrena-2023": {
        "desc": "A hybrid approach for solving the gravitational N-body problem with Artificial Neural Networks. Abstract: stract Simulating the evolution of the gravitational N -body e problem becomes extremely com- putationally expensive as N increases since the problem r complexity scales quadratically with the number of bodies. In order to alleviate this problem, we study the use of Artificial Neural Networks (ANNs) to replace expensive partsr of the integration of planetary systems. Neural networks that include physical knowledge have rapidly grown in popularity in the last few motion years, although of celestial few bodies. attempts For have this purpose, been made we e to study use them the advantages to speed up and the limitations simulation of of using the simulation Hamiltonian ofNeural planetary Networks systems. to replace We e compare computationally the results expensive of the numerical parts of integration the numerical of and a planetary a conventional system Deep with Neural asteroids Network. with p those Due obtained to the non-linear by a Hamiltonian nature of the Neural gravitational Network equations of motion, errors in the integration propagate, which may lead to divergence from the reference solution. To t increase the robustness of a method that uses neural networks, with we propose the numerical a hybrid solution integrator o if considered that evaluates inaccurate.  the prediction of the network and replaces it Hamiltonian Neural Networks can make predictions that resemble the behavior of sym- orders plectic of integrators magnitude. but In n are contrast, challenging Deep to Neural train and Networks in our case are fail easy when to train the inputs but fail differ to con- ∼7 serve energy, leading to fast divergence from the reference solution. The hybrid integrator designed to include t the neural networks increases the reliability of the method and prevents hand, large energy the n use errors of neural without networks increasing results the in computing faster simulations cost significantly. when the number For the of problem asteroids at is &70.",
        "refs": "",
    },
    "xiong-2022": {
        "desc": "Nonseperable Symplectic Neural Networks. Predicting the behaviors of Hamiltonian systems has been drawing increasing attention in scientific machine learning. However, the vast ma- jority of the literature was focused on predicting separable Hamiltonian systems with their kinematic and potential energy terms being explicitly decoupled while building data-driven paradigms to predict nonseparable Hamiltonian systems that are ubiquitous in fluid dynamics and quantum mechanics were rarely explored. The main computational challenge lies in the effective embedding of symplectic priors to describe the inherently coupled evolution of position and momentum, which typically exhibits intricate dynamics. To solve the problem, we propose a novel neural net- work architecture, Nonseparable Symplectic Neural Networks (NSSNNs), to uncover and embed the symplectic structure of a nonseparable Hamil- tonian system from limited observation data. The enabling mechanics of our approach is an augmented symplectic time integrator to decouple the position and momentum energy terms and facilitate their evolution. We demonstrated the efficacy and versatility of our method by predicting a wide range of Hamiltonian systems, both separable and nonseparable, in- cluding chaotic vortical flows. We showed the unique computational mer- its of our approach to yield long-term, accurate, and robust predictions for large-scale Hamiltonian systems by rigorously enforcing symplectomor- phism.",
        "refs": "",
    },
    "zhang-2012": {
        "desc": "Universal Approximation of Extreme Learning Machine with Adaptive Growth of Hidden Nodes. Abstract: Extreme learning machines (ELMs) have been proposed for generalized single-hidden-layer feedforward net- works which need not be neuron-like and perform well in both regression and classification applications. In this brief, we propose an ELM with adaptive growth of hidden nodes (AG-ELM), which provides a new approach for the automated design of networks. Different from other incremental ELMs (I-ELMs) whose existing hidden nodes are frozen when the new hidden nodes are added one by one, in AG-ELM the number of hidden nodes is determined in an adaptive way in the sense that the existing networks may be replaced by newly generated networks which have fewer hidden nodes and better generalization performance. We then prove that such an AG-ELM using Lebesgue p−integrable hidden activation func- tions can approximate any Lebesgue p−integrable function on a compact input set. Simulation results demonstrate and verify that this new approach can achieve a more compact network architecture than the I-ELM.",
        "refs": "",
    },

    # BOOKS



    # TODO

    # RANDOM FEATURE MODELS
    "random feature model": {
        "desc": "",
        "refs": "",
    },
    # DATA AGNOSTIC DISTRIBUTION
    "data agnostic distribution": {
        "desc": "",
        "refs": "",
    },
    # DATA DRIVEN DISTRIBUTION
    "data driven distribution": {
        "desc": "",
        "refs": "",
    },
    # RANDOM FOURIER FEATURES
    "random fourier features": {
        "desc": "",
        "refs": "",
    },
    # NEURAL OPERATOR
    "NEURAL OPERATOR": {
        "desc": "",
        "refs": "",
    },
    # FOURIER NEURAL OPERATOR
    "FOURIER NEURAL OPERATOR": {
        "desc": "",
        "refs": "",
    },


}


def main():
    parser = argparse.ArgumentParser(description='Definitions learned in thesis. Always input with singular terms, and also always write uncapitalized, and separate words with space.')

    parser.add_argument('key', type=str, help='Definition input')

    args = parser.parse_args()
    key = args.key

    if not key in defs:
        print('KEY NOT FOUND! Would you like to add it maybe?')
        exit(1)

    print('\n')
    print('KEY: ' + str(key))
    print('DESCRIPTION: ' + defs[key]["desc"])
    print('REFERENCES : ' + defs[key]["refs"])
    print('\n')

if __name__ == "__main__":
    main()
