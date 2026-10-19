Clustering is quite new to the field of quantum computing but it may bring some interesting techniques worth testing out. Quantum clustering does not differ too much from the idea of finding blocks of data close to each other considering the dimensional space composed by the features. Therefore, we would need to find a good grouping strategy or proximity measure to evaluate the closeness between samples.

Falcondale offers three methods, all of them scikit-learn estimators living in `falcondale.clustering`:

| Method | Estimator | Idea | Needs |
|--------|-----------|------|-------|
| `pqc` | `QuantumClustering` | Probabilistic Quantum Clustering, fuzzy memberships from a quantum potential | `pip install "falcondale[clustering]"` (APQC and TensorFlow) |
| `qc` | `SchrodingerClustering` | The original Quantum Clustering, samples roll down a quantum potential | Core install |
| `qkmeans` | `QuantumKMeans` | k-means or k-medians with distances estimated by quantum interference | Core install |

## Probabilistic Quantum Clustering

This promising technique by [Casaña-Eslava et al.](https://arxiv.org/abs/1902.05578) shows a quantum inspired approach able to find fuzzy membership regions based on quantum formalism. Similar to density based clustering the core idea is to be able to map a function $\psi$ so that it reveals the membership on a given sample.

<figure markdown>
  ![membershipfunc.png](../assets/membershipfunc.png){ width="500" }
  <figcaption>Membership function representation as wave functions</figcaption>
</figure>

Every sample descends the quantum potential built from $\psi$ until it reaches a well, and samples sharing a well form a cluster. Nearby wells can then be merged hierarchically according to the energy barrier between them, which gives coarser partitions.

### Choosing the bandwidth

One key parameter of this technique is the bandwidth $\sigma$ of the membership function. This is a common feature in many other implementations (number of clusters in K-means or eps value on DBSCAN). Too small and every sample ends up in its own well, too large and everything merges into a single cluster.

Instead of asking you for the right value, `QuantumClustering` searches a grid of bandwidths. Each one can be:

* `"knn:<fraction>"`, an adaptive bandwidth per sample given by the distance to its nearest neighbours (a fraction of the dataset),
* `"scalar:<sigma>"`, a single bandwidth in the normalised coordinates APQC works with.

Each bandwidth produces candidate partitions: the raw wells (`base`) and, if `merge=True`, each level of the energy merge (`energy_1`, `energy_2`,...). Candidates are ranked by their training [silhouette](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient), and only *admissible* ones can be selected: between 2 and 8 clusters, the smallest one holding at least `max(3, 3%)` of the samples. Every candidate is kept in `candidates_`, so you can check why a partition was chosen.

### Train and test

`QuantumClustering` follows the scikit-learn API. It can be fitted on a train split and then assign unseen samples together with their membership probabilities. Preprocessing should also be fitted on the train split only, so that no information from the test samples leaks into the model. `make_clustering_pipeline` builds a scikit-learn pipeline with median imputation, standard scaling, PCA and the quantum clustering step that does just that.

```py
from sklearn.model_selection import train_test_split
from falcondale.clustering import make_clustering_pipeline

train, test = train_test_split(data, test_size=0.3, random_state=101)

pipeline = make_clustering_pipeline(components=6, random_state=101)
pipeline.fit(train)

labels = pipeline.predict(test)
probabilities = pipeline.predict_proba(test)  # one column per cluster
```

The fitted estimator is the last step of the pipeline and holds the details of the selected partition.

```py
import pandas as pd

model = pipeline[-1]
model.bandwidth_, model.stage_, model.n_clusters_
pd.DataFrame(model.candidates_)
```

The probabilities are the algorithmic membership weights of APQC, not calibrated probabilities, but they are useful to tell confident assignments from samples lying between two clusters.

Other options can be tuned when creating the estimator: the bandwidth grid (`bandwidths=["knn:0.2", "scalar:0.4"]`), the descent budget (`steps`, `patience`, `learning_rate`) and the admissibility rules (`min_clusters`, `max_clusters`, `min_cluster_size`, `min_cluster_fraction`). Setting `random_state` makes the results reproducible.

### Within a project

Clustering is an unsupervised task, so a project can be created without a target. `cluster` fits the whole (preprocessed) dataset and returns the label of each sample and its probability.

```py
from falcondale import Project

myproject = Project(dataset)
myproject.preprocess()

label, proba = myproject.cluster(ctype="pqc", random_state=101)
```

A single bandwidth can still be forced with `sigma`, for example `myproject.cluster(ctype="pqc", sigma=0.4)`.

## Quantum Clustering by Horn and Gottlieb

Probabilistic Quantum Clustering builds on an older idea by [Horn and Gottlieb](http://horn.cs.tau.ac.il/publications/nipsQC.pdf). Take a wave function made of a Gaussian of width $\sigma$ around every sample,

$$
\psi(x) = \sum_i e^{-\frac{|x - x_i|^2}{2\sigma^2}}
$$

and ask it to be the ground state of a Schrödinger equation. The potential that makes it so is

$$
V(x) = E - \frac{d}{2} + \frac{1}{2\sigma^2 \psi(x)} \sum_i |x - x_i|^2 e^{-\frac{|x - x_i|^2}{2\sigma^2}}
$$

Where samples are dense the potential has minima, which play the role of cluster centres. Each sample rolls down the potential by gradient descent and samples ending at the same minimum form a cluster, so there is no need to tell how many clusters to find. The width $\sigma$ still decides how fine the partition is.

`SchrodingerClustering` only needs NumPy and SciPy. The widths are given relative to the spread of the data (the root mean square of the feature standard deviations), and, as `QuantumClustering` does with its bandwidths, it tries a grid of them and keeps the admissible partition with the best training silhouette.

```py
from falcondale.clustering import SchrodingerClustering, make_clustering_pipeline

pipeline = make_clustering_pipeline(components=6, estimator=SchrodingerClustering(sigmas=[0.5, 0.75, 1.0]))
pipeline.fit(train)

labels = pipeline.predict(test)  # test samples roll down the train potential
weights = pipeline.predict_proba(test)  # share of the wave function coming from each cluster
```

`pipeline[-1].candidates_` lists every width tried and `cluster_centers_` holds the minima found.

## Quantum k-means

The best known clustering algorithm can also be brought to quantum computers. In k-means the expensive step is computing the distance between every sample and every center, and a quantum computer can estimate it by interference ([Khan et al.](https://arxiv.org/abs/1909.12183), [Poggiali et al.](https://arxiv.org/abs/2212.06691)).

A sample $u$ and a center $v$ are amplitude encoded together, $|\psi\rangle = \frac{1}{\sqrt{Z}}(|0\rangle|u\rangle + |1\rangle|v\rangle)$ with $Z = |u|^2 + |v|^2$, which only needs $\lceil \log_2 d \rceil + 1$ qubits for $d$ features. A Hadamard gate on the first qubit makes both halves interfere, and the probability of measuring it as a one is

$$
P(1) = \frac{|u - v|^2}{2Z}
$$

so the squared distance is $2 Z P(1)$. Measuring a finite number of times (`shots`) gives a noisy estimate, just as a real device would.

```py
from falcondale.clustering import QuantumKMeans

pipeline = make_clustering_pipeline(components=6, estimator=QuantumKMeans(n_clusters=4, shots=1024, random_state=101))
pipeline.fit(train)

labels = pipeline.predict(test)
distances = pipeline.transform(test)  # estimated distance to each center
```

Centers are updated classically. With `center="median"` the geometric median of each cluster is used instead of the mean, the quantum k-medians [used at CERN to find anomalous collision events](https://arxiv.org/abs/2301.10780), which is less sensitive to outliers. In that kind of problem the distance to the closest center, `transform(X).min(axis=1)`, is a natural anomaly score.

The circuits are simulated, so this is about learning how the algorithm works rather than speed. To run one on real hardware, `falcondale.helpers.interference.interference_circuit(u, v)` returns the Qiskit circuit for a pair of vectors (it needs `falcondale[clustering]`).

## From a project

All methods can be called from a project, which clusters the whole (preprocessed) dataset and returns the label of each sample with its probability or membership weight:

```py
labels, proba = myproject.cluster(ctype="qc")
labels, membership = myproject.cluster(ctype="qkmeans", n_clusters=3, center="median")
```

There is plenty of room for improvement on this particular topic, for example with variational approaches. As a reference take this work from the people at [Multiverse Computing](https://www.nature.com/articles/s41598-023-39771-6).
