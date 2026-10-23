"""
Falcondale SDK: quantum machine learning for computer scientists.

    >>> from falcondale import Project
    >>> project = Project(dataframe, target="target")
"""

from importlib.metadata import version

from falcondale.assessment import QMLAssessor
from falcondale.clustering import QuantumClustering
from falcondale.data import Dataset
from falcondale.exceptions import TooManyQubitsNeeded
from falcondale.model import Model
from falcondale.project import Project

__all__ = ["Dataset", "Model", "Project", "QMLAssessor", "QuantumClustering", "TooManyQubitsNeeded"]
__version__ = version("falcondale")
