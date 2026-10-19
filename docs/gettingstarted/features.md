# Features

Falcondale aims to make Quantum Machine Learning easy to use but also highly adaptable to those with specific needs or willing to test different options. At current version it provides:

* Simple and easy **data preprocessing** for the purpose. You can bring your own data engineering practices but we also want to make it easy for those new to the field
* **Quantum Feature Selection** using _state-of-the-art_ techniques: simulated annealing, simulated bifurcation and QAOA
* **Quantum Classification** models with a diverse set of options: Quantum Support Vector Machines and Quantum Neural Networks, together with a classical baseline to compare against
* **Quantum Clustering** by means of quantum inspired approaches like Probabilistic Quantum Clustering, with a train/test workflow and membership probabilities for unseen samples

All these features count with a broad set of options so that each one can be customized and enable benchmarking. It also brings together some industry widely adopted frameworks so that contributions to these can be included as the usage grows.

* IBM's [Qiskit framework](https://qiskit.org/)
* Xanadu's [Pennylane framework](https://pennylane.ai/) for quantum differential programming
* D-Wave's open source [dimod](https://github.com/dwavesystems/dimod) and [dwave-samplers](https://github.com/dwavesystems/dwave-samplers)
* [Simulated Bifurcation](https://github.com/bqth29/simulated-bifurcation-algorithm) on top of PyTorch

Everything runs locally on your laptop, no account or token is needed.

Future releases could also count with integrations so that users can select the target device in which their algorithms would run, such as IBM Quantum or other cloud providers. In the meantime, the QUBO and Ising formulations used for feature selection are exposed so that you can send them to any device you have access to.

This is a collaborative effort so we hope a growing list of integrations can be brought to Falcondale with the help of everyone!
