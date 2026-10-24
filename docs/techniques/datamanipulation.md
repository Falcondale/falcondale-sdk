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

For a closer look, `table_report()` builds an interactive [skrub](https://skrub-data.org/) report with the distribution of every column, missing values and how columns relate to each other. It renders in notebooks and `.open()` shows it in the browser.

```py
myproject.table_report()
```

## Preprocessing

In order to ease the work of the classifiers some preprocessing steps need to be implemented. Simple things like filling NAs or some more complex techniques like dimensionality reduction can be performed. This is an important step as filling all cells, standardizing the data and reducing dimensions will ease the execution when going into real hardware. Moreover, simulating systems beyond 20 variables (mapped to 20 qubits in general) would start to consume quite some resources on your machine. That is why, whenever possible, dimensionality reduction will help speed up our analysis.

```py
myproject.preprocess()
```

Is the most simple command we can send and will basically fill missing data with zeros and scale the features into the $[0, 1]$ range. This can be checked by calling to the following function.

```py
myproject.show_features()
```

## Categories, text and dates

Real tables rarely come all numeric. Customer plans, cities, job titles or signup dates need to be turned into numbers first, and quantum models make that choice a bit more delicate than usual: every new feature usually costs a qubit.

When the dataset has non numeric columns, `preprocess()` encodes them with skrub's [TableVectorizer](https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html), which needs the `tabular` extra (`pip install "falcondale[tabular]"`). Each column is treated according to its kind:

| Kind | Example | Encoding |
|------|---------|----------|
| Numeric | `tenure` | Kept as it is |
| Category | `plan` with 3 values | One hot encoded, one feature per value (`plan_basic`, `plan_plus`,...) |
| Text | `job` with 40 titles | Compressed into a few components (`job_0`, `job_1`,...) |
| Date | `signup` | Split into numeric parts (`signup_year`, `signup_month`,...) |
| Constant or empty | `country` always `ES` | Dropped |

Then missing values are filled with the median of each feature, missing categories get their own indicator (`city_nan`) and, as usual, everything is scaled into $[0, 1]$.

The defaults are more compact than skrub's, as we care about the number of features: columns with fewer than 10 distinct values are one hot encoded and the others are compressed into 4 components. Both can be changed, as well as any other `TableVectorizer` argument.

```py
myproject.preprocess(cardinality_threshold=5, string_components=2)
```

`encode=True` applies the same treatment (median imputation included) to all numeric tables, while `encode=False` refuses to encode and raises an error if there is anything to encode. The fitted encoder travels with the trained models, so `model.predict(new_data)` accepts raw rows, unseen categories and missing values included.

If you want to know in advance how many features each column will become, which columns are redundant or not related to the target at all, the [QML assessment](assessment.md#looking-at-the-raw-table) diagnoses the raw table too.

## Dimensionality reduction

Falcondale supports to call [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) based dimensionality reduction at the moment by simply invoking the method indicating the amount of dimensions to be reduced to.

```py
myproject.preprocess(reduced_dimension=3)
```
Principal components are added as new features named `pca0`, `pca1`,... and selected for the following steps. This method could be extended in the future by adding more complex techniques for embedding data into lower dimensional spaces.
