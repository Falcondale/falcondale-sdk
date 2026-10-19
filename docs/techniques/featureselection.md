One of the key use cases for Quantum Computing over Classical resources is their capacity to explore the vast set of options that combinatorial optimization problems may pose. Feature Selection deals with finding a subset of the features in the original dataset so that by using a subset a more accurate model with better generalization capacities is obtained. Basically removes the noise or redundant information from the original dataset by removing specific features. It also benefits cases where certain features are hard to obtain because they might involve expensive or lengthy computation (examples can be from clinical analysis datasets when it comes to patient data, large historic datasets where one time features need to be found, etc.).

There are different techniques we may to obtain this subset but most of them require training models to check if the selection has been improved or not. This is the case for [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html). Others do it by simply removing data with [low variance](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold).

Quantum Computers do this differently. They focus on the informative capacity of the dataset. If a feature is good detecting a given label this has higher chances to remain; if two features show similarities in their behavior and provide similar prediction capacity over the target label then we may be able to suppress one of those. Therefore, we can encode our problem into a QUBO form and use a quantum computer to solve this combinatorial optimization problem. Quantum annealers among others are known to be good at this tasks.

<figure markdown>
  ![featureselection.png](../assets/featureselection.png)
  <figcaption>Feature selection mapped to a QUBO problem</figcaption>
</figure>

More in detail about the mathematical formulation involved can be found [in this work](https://arxiv.org/pdf/2203.13261.pdf) and in the [mRMR formulation by Mücke et al.](https://doi.org/10.1007/s42484-023-00099-z) described below.

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

The natural option to solve our QUBO formulation is to use a Quantum Annealing option, but this problem can also be tackled by digitized computers, mostly using variational techniques such as QAOA. One of the main challenges when approaching variational techniques is simulating a complex evolution while iterating over its parametrization. Simulating $n$ qubits requires keeping $2^n$ amplitudes in memory, so Falcondale limits QAOA to 20 features and warns you beyond 12. A relevance pre-screen (`pool`, see below) or a PCA reduction will help you get there.

```python
myproject.feature_selection(max_cols=3, method="qaoa")
```

Behind the scenes, Falcondale builds a QUBO and maps it to an Ising Hamiltonian, optimizes the QAOA circuit parameters using a statevector simulator and samples the resulting circuit, keeping the bitstring with the lowest energy. Options such as `reps`, `shots` or `maxiter` can be passed along.

Finally, small problems can be solved exactly. The `exact` method evaluates every selection of `max_cols` features, 230 thousand of them for 6 out of 26 features, which NumPy goes through in about a second. It is the reference any quantum or quantum-inspired solver should be compared against.

```python
myproject.feature_selection(max_cols=3, method="exact")
```

### Choosing a solver

| Method | Technique | Runs on | Good for |
|--------|-----------|---------|----------|
| `sa` | Simulated Annealing | CPU | A robust default for any size |
| `sb` | Simulated Bifurcation | CPU or GPU (PyTorch) | Large problems |
| `qaoa` | Quantum Approximate Optimization Algorithm | Statevector simulator | Learning how gate based quantum optimization works, up to 20 features |
| `exact` | Enumeration of every selection | CPU | The optimal selection when there are few candidates, and a reference for the rest |

All of them can be seeded (`seed=42`) so that results are reproducible. Heuristic solvers may stop short of the optimal selection, since swapping a feature means crossing the penalty that keeps exactly `max_cols` of them. More effort (`num_reads` or `num_sweeps` for `sa`, `agents` for `sb`) usually gets them closer.

### Fewer candidates first

When there are more features than a solver can handle, as it happens with QAOA beyond 20 qubits or with the exact solver, the `pool` argument keeps only the most relevant ones before building the QUBO.

```python
myproject.feature_selection(max_cols=3, method="qaoa", pool=12)
```

Relevance is the mutual information between each feature and the target, and it can be inspected directly.

```python
from falcondale.feature_selection import relevance

relevance(myproject.dataset)  # a pandas Series, from the most to the least relevant feature
```

### Bring your own solver

The QUBO itself is available, so it can be shipped to any solver or quantum hardware provider you have access to.

```py
from falcondale.feature_selection import build_qubo

bqm = build_qubo(myproject.dataset, max_cols=3, formulation="mrmr")
```

`bqm` is a [dimod](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/) `BinaryQuadraticModel`, the common language of most annealing and QUBO solvers. Its Ising Hamiltonian counterpart for gate based devices can be obtained with `falcondale.feature_selection.ising_operator(bqm)`.

### How information is measured

Both relevance and redundancy are estimated from histograms of the data, and those estimates are biased. A column split into many cells always looks a bit informative, even if it is pure noise, simply because a finite sample never fills all cells evenly. This bias grows with the number of cells and shrinks with the number of samples.

To keep it under control, every variable (target included) is discretized into at most 5 bins by default. Binary and ordinal columns with few levels keep them, while the rest are split by quantiles so that every bin holds a similar number of samples. Repeated values are never split, so a skewed column such as days in hospital, mostly zeros, gets one bin for the zeros and the remaining bins for the tail. Continuous targets are treated the same way, which makes regression problems work too.

```python
myproject.feature_selection(max_cols=3, bins=5, strategy="quantile")
```

`strategy="uniform"` uses equal width bins instead, the behaviour of previous releases (`bins=10`).

### Two QUBO formulations

The linear terms of the QUBO always reward relevant features, $-I(x_i; y)$. The quadratic terms are what tell the formulations apart.

**mRMR** (minimum redundancy, maximum relevance) is the default. Following [Mücke et al.](https://doi.org/10.1007/s42484-023-00099-z), pairs of features sharing information are penalized, averaged over the $k - 1$ pairs a feature has in a selection of $k$ features:

$$
E(z) = -\sum_i I(x_i; y)\, z_i + \frac{1}{k - 1} \sum_{i<j} I(x_i; x_j)\, z_i z_j
$$

It only needs pairwise histograms, which are reliable even with a few hundred samples.

**CMI** rewards pairs of features whose information about the target is complementary, using the conditional mutual information $I(x_i; y | x_j) + I(x_j; y | x_i)$ as a negative quadratic term. It can capture interactions mRMR misses, but it needs histograms of the target and two features together. With 5 bins that is 125 cells, so it needs many more samples than mRMR to be trustworthy.

```python
myproject.feature_selection(max_cols=3, formulation="cmi")
```

As a rule of thumb, use `mrmr` unless you have thousands of samples and reasons to expect features that only matter in combination. On a synthetic cohort with six planted causal factors, their proxies and noise (`tests/data/risk_factors.csv`), the exact solver gives the following number of causal factors among the 6 selected ones:

| Outcome | mRMR, quantile bins | CMI, quantile bins | CMI, 10 equal width bins (previous releases) |
|---------|---------------------|--------------------|----------------------------------------------|
| Binary | 5 | 4 | 2, plus the 17 level noise column |
| Continuous | 6 | 5 | 2, plus the 17 level noise column |

Most of the gain comes from the discretization, and mRMR adds a bit more on top.

In both formulations a penalty term $\lambda (\sum_i z_i - k)^2$ forces the selection of exactly $k$ features. Its strength $\lambda$ is computed from the data, just large enough for the lowest energy to always be a selection of $k$ features.
