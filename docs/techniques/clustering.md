Clustering is quite new to the field of quantum computing but it may bring some interesting techniques worth testing out. Quantum clustering does not differ too much from the idea of finding blocks of data close to each other considering the dimensional space composed by the features. Therefore, we would need to find a good grouping strategy or proximity measure to evaluate the closeness between samples.

Quantum clustering relies on the [APQC](https://pypi.org/project/apqc/) package and TensorFlow, which are not installed by default. Install them with:

```
pip install "falcondale[clustering]"
```

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

There is plenty of room for improvement on this particular topic with implementations on purely quantum approaches in the following releases. As a reference take this work from the people at [Multiverse Computing](https://www.nature.com/articles/s41598-023-39771-6).
