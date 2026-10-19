Taking it one step further, Quantum Machine Learning allows us to expand the classical models we may be using in our common practice to a new level. Falcondale  enables some of the most used QML models for classification in a user-friendly way.

For the time being Falcondale supports mostly binary classification models so most of the examples below will dig into the most mature techniques for this but we are constantly researching to include new models allowing for more complex options (multiclass problems, time-series forecasting,...), so stay tuned!

For models explained below many different implementations can be found, by simply changing the *feature map* used to encode classical data into quantum states or the inner architecture of the QNN model (ansatz) a huge amount of options can appear. Falcondale aims to implement internally several mechanisms to find the best possible option so that novice users will not have to struggle with this but, of course, the advanced user may be able to select between different options. In the long term the _auto_ feature should enable anyone to benchmark between all potential options without consuming too many quantum resources.

### Quantum Support Vector Classifier (QSVC)

QSVC is one of the most mature QML models we can encounter. It goes back to 2014 when this model was proposed ([article](https://arxiv.org/abs/1307.0471)). The basic idea is that our quantum circuit can be interpreted as a mapping function that will compute the kernel between samples defined as

$$
K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|².
$$

Now, how do we find $\phi$ ? This is the tricky part, we would need to look for a suitable quantum circuit (known as feature map) that maps our data into a Hilbert space, ideally maximizing the separation between samples of the two classes.

Well, Falcondale has you covered on that. By invoking:

```py
model = myproject.evaluate("qsvc")
model.print_report()
```

We would obtain in the `model` variable the object associated to the training of the model, splitting target training data into two sets so that train/test splitting is also present when providing training performance data.

This return object belongs to the Falcondale Model class and contained, apart from the model relevant information so that it can all be contained within a single object. Performance metrics can be printed by asking the available metrics or simply by invoking the _print_report()_ method.

```py
              precision    recall  f1-score   support

         0.0       0.95      0.99      0.97        93
         1.0       0.99      0.93      0.96        71

    accuracy                           0.96       164
   macro avg       0.97      0.96      0.96       164
weighted avg       0.96      0.96      0.96       164
```

We can change the ratio to be used for validation of the model training by selecting a test size ratio.

```py
model = myproject.evaluate("qsvc", test_size=0.3)
```

Two implementations are available. By default the [Qiskit](https://qiskit.org/) one uses a ZZ feature map, while the [PennyLane](https://pennylane.ai/) one uses angle embedding.

```py
model = myproject.evaluate("qsvc", backend="pennylane")
```

Both of them need one qubit per feature and compute the kernel from simulated statevectors, so up to 16 features are allowed. Use feature selection or dimensionality reduction to get there.

### A classical baseline

It is always a good idea to compare against a classical model trained on the same data. The `svc` option trains a classical Support Vector Classifier, so you can check whether the quantum model brings anything to the table.

```py
baseline = myproject.evaluate("svc")
baseline.metric("auc")
```

### Quantum Neural Networks (QNN)

Quantum Neural Networks is basically the idea of merging the ability to use layered parameterized structures coming from classical Neural Networks with the concept of Parameterized Quantum Circuits (PQC). This way, we can set up a PQC or ansatz that can be tuned towards a classification purpose.

Doing a sensible ansatz choice can be tricky, but this is also part of Falcondale’s secret sauce. Just by invoking:

```python
model = myproject.evaluate("qnn", test_size=0.3)
```

A layered parameterized quantum circuit will be suited for the particular characteristics of the provided dataset. This is a non trivial task as pointed out by [Larocca et al.](https://arxiv.org/abs/2109.11676) but there is room to experiment in the current setup, by selecting a number of layers that will affect the final architecture of the ansatz.

```python
model = myproject.evaluate("qnn", test_size=0.3, layers=4, epochs=30, learning_rate=0.05)
```

The training progress can be printed by passing `verbose=True`.

Current implementation performs [Amplitude Embedding](https://pennylane.ai/qml/glossary/quantum_embedding/), so $n$ features only need $\lceil \log_2 n \rceil$ qubits, followed by strongly entangling layers. We would like to extend this so that advanced users can try out different setups for both data embedding as well as layer structures.

### Using the models outside Falcondale

Every model follows the [scikit-learn](https://scikit-learn.org/) estimator API. The trained estimator is available as `model.estimator`, and the classes can be used directly in your own pipelines.

```python
from falcondale.helpers.qnn import QNNClassifier
from sklearn.model_selection import cross_val_score

cross_val_score(QNNClassifier(layers=2), X, y, cv=3)
```

### Report and metrics

In order to provide as much information as possible for the outcome of the project, Falcondale provides a variety of metrics out of the box.

- **Classification report**

    This is a complete report on the outcome of the model evaluating our tests samples after model has been trained. It shows the precision (ratio of true positives over all flagged positives), recall (true positive ratio over actual positive samples) and f1-score as well as the amount of samples that supported those metrics per label/class.

    Accuracy might be a good metric in those cases where dataset is balanced, but if not high discrepancies will be seen between accuracy and balanced accuracy as the model is biased towards the majority class.

    This report aims to provide a set of metrics needed in order to evaluate the model on its full extent. When a model is trained by default the ```print_report()``` method shows the whole classification report but additional metrics can be found.

- **AUC**

    Additionally AUC, short for **a**rea **u**nder the ROC (receiver operating characteristic) **c**urve, is also provided. AUC is an average of true positive rates over all possible values of the false positive rate and it provides a sensible way to evaluate the bias of the model as well as its potential for error when put into operation.

- **Additional metrics**

    Taken from the confusion matrix generated after model being trained and tested against the test dataset, precision, recall, specificity, balanced accuracy, accuracy and f1 score among other metrics can be directly consulted by requesting the available metrics to the model (using the ```list_metrics()``` function).

```python
# List available metrics
model.list_metrics()

# Select one
model.metric("auc")
```

## Prediction

Now that the model has been trained, you might wonder how it can be used. Well, that is why model was returned after each training job was finished. This object ensures same preprocessing mechanisms are followed when evaluating a new sample set.

Therefore, few steps are needed in order to evaluate a new dataset. First, inform of its existence and the purpose of it.

```py
y_pred = model.predict(X)
```

The model will return the target class so that users can evaluate or use its response for potentially longer or more complex jobs where Falcondale is introduced as an intermediary step.

This is often recommended following Falcondale's philosophy, making Quantum Machine Learning not a substitution but an addition to the current model toolbox a company may have.

Models can be stored and loaded back. As with any pickle file, only load models you trust.

```py
model.save("my_model.pkl")

from falcondale import Model

model = Model.load("my_model.pkl")
```
