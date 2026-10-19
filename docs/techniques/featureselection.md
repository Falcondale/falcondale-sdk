One of the key use cases for Quantum Computing over Classical resources is their capacity to explore the vast set of options that combinatorial optimization problems may pose. Feature Selection deals with finding a subset of the features in the original dataset so that by using a subset a more accurate model with better generalization capacities is obtained. Basically removes the noise or redundant information from the original dataset by removing specific features. It also benefits cases where certain features are hard to obtain because they might involve expensive or lengthy computation (examples can be from clinical analysis datasets when it comes to patient data, large historic datasets where one time features need to be found, etc.).

There are different techniques we may to obtain this subset but most of them require training models to check if the selection has been improved or not. This is the case for [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html). Others do it by simply removing data with [low variance](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold).

Quantum Computers do this differently. They focus on the informative capacity of the dataset. If a feature is good detecting a given label this has higher chances to remain; if two features show similarities in their behavior and provide similar prediction capacity over the target label then we may be able to suppress one of those. Therefore, we can encode our problem into a QUBO form and use a quantum computer to solve this combinatorial optimization problem. Quantum annealers among others are known to be good at this tasks.

<figure markdown>
  ![featureselection.png](../assets/featureselection.png)
  <figcaption>Feature selection mapped to a QUBO problem</figcaption>
</figure>

More in detail about the mathematical formulation involved can be found [in this work](https://arxiv.org/pdf/2203.13261.pdf).

The tricky thing comes when this QUBO formulation needs to be embedded into a quantum hardware. Instead of doing all this manually, Falcondale offers the ability to simply call:

```python
myproject.feature_selection(max_cols=3)
```

It will trigger a feature selection workflow targeting *survived* feature in above example for the Titanic dataset. By default, and if no additional information is provided, it will try to solve it using Simulated Annealing provided by D-Wave's open source [dwave-samplers](https://github.com/dwavesystems/dwave-samplers) package, running on your own machine. If we think the problem is large enough or simulated annealing cannot provide the solution close to what we would like, we can select different modalities for it to be solved, such as **simulated bifurcation**. Simulated bifurcation is a technique known to work even with large problems, such as a 2000 spin MAX-CUT problem as shown by the original work by [Goto et al.](https://www.science.org/doi/10.1126/sciadv.aav2372).

```python
myproject.feature_selection(max_cols=3, method="sb")
```
By default it uses the ballistic approach that should converge faster but might be prone to local minima instead of global solutions, as described in the original work by [Kanao et al.](https://www.nature.com/articles/s42005-022-00929-9). Additional options are passed to the solver, so heated or discrete SB can be tried too.

```python
myproject.feature_selection(max_cols=3, method="sb", mode="discrete", heated=True, seed=42)
```

The natural option to solve our QUBO formulation is to use a Quantum Annealing option, but this problem can also be tackled by digitized computers, mostly using variational techniques such as QAOA. One of the main challenges when approaching variational techniques is simulating a complex evolution while iterating over its parametrization. Simulating $n$ qubits requires keeping $2^n$ amplitudes in memory, so Falcondale limits QAOA to 20 features and warns you beyond 12. A previous selection with `sa` or a PCA reduction will help you get there.

```python
myproject.feature_selection(max_cols=3, method="qaoa")
```

Behind the scenes, Falcondale builds a QUBO and maps it to an Ising Hamiltonian, optimizes the QAOA circuit parameters using a statevector simulator and samples the resulting circuit, keeping the bitstring with the lowest energy. Options such as `reps`, `shots` or `maxiter` can be passed along.

### Choosing a solver

| Method | Technique | Runs on | Good for |
|--------|-----------|---------|----------|
| `sa` | Simulated Annealing | CPU | A robust default for any size |
| `sb` | Simulated Bifurcation | CPU or GPU (PyTorch) | Large problems |
| `qaoa` | Quantum Approximate Optimization Algorithm | Statevector simulator | Learning how gate based quantum optimization works, up to 20 features |

All of them can be seeded (`seed=42`) so that results are reproducible.

### Bring your own solver

The QUBO itself is available, so it can be shipped to any solver or quantum hardware provider you have access to.

```py
from falcondale.feature_selection import build_qubo

bqm = build_qubo(myproject.dataset, max_cols=3)
```

`bqm` is a [dimod](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/) `BinaryQuadraticModel`, the common language of most annealing and QUBO solvers. Its Ising Hamiltonian counterpart for gate based devices can be obtained with `falcondale.feature_selection.ising_operator(bqm)`.
