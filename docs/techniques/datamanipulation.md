How data is treated is really important in all Data Science projects and it is also the case for Quantum Machine Learning. Due to the techniques used we need all data to be informed, make it numerical and with few but informative features to be used.

That is why Data Manipulation is a relevant step in order to proceed with any other technique within the Falcondale package.

Trying to make it as easy as possible to the users, the creation of a Project requires indicating which feature is the target variable of our exercise.

```py
from falcondale import Project

myproject = Project(dataset, target="target")
```

Before going any further, it is a good idea to profile the dataset. Falcondale will point out common issues such as missing values, duplicated rows, non numeric or constant columns and how balanced the target is.

```py
myproject.profile_dataset()
```

A report similar to the following one will be shown

```
Number of samples on the dataset: 569
Number of columns on the dataset: 31
target has been selected as target variable for supervised learning tasks.
Target class balance: 1: 62.7%, 0: 37.3%
Ratio of cells with missing values: 0.00%
All data is numeric therefore little preprocessing might need to be done.
```

The same information is returned as a dictionary, in case you want to use it programmatically.

## Preprocessing

In order to ease the work of the classifiers some preprocessing steps need to be implemented. Simple things like filling NAs or some more complex techniques like dimensionality reduction can be performed. This is an important step as filling all cells, standardizing the data and reducing dimensions will ease the execution when going into real hardware. Moreover, simulating systems beyond 20 variables (mapped to 20 qubits in general) would start to consume quite some resources on your machine. That is why, whenever possible, dimensionality reduction will help speed up our analysis.

```py
myproject.preprocess()
```

Is the most simple command we can send and will basically fill missing data with zeros and scale the features into the $[0, 1]$ range. This can be checked by calling to the following function.

```py
myproject.show_features()
```

## Dimensionality reduction

Falcondale supports to call [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) based dimensionality reduction at the moment by simply invoking the method indicating the amount of dimensions to be reduced to.

```py
myproject.preprocess(reduced_dimension=3)
```
Principal components are added as new features named `pca0`, `pca1`,... and selected for the following steps. This method could be extended in the future by adding more complex techniques for embedding data into lower dimensional spaces.
