"""
Helpers for the optional dependency families (extras) of Falcondale.
"""

from collections.abc import Generator
from contextlib import contextmanager

EXTRAS = {
    "feature-selection": "quantum feature selection",
    "qml": "quantum classifiers",
    "clustering": "quantum clustering",
}


@contextmanager
def optional_dependencies(extra: str) -> Generator[None]:
    """
    Turns an ImportError raised inside the block into a message telling which extra to install.

    Args:
        extra (str): Name of the extra providing the dependencies
    """
    try:
        yield
    except ImportError as err:
        raise ImportError(
            f"{EXTRAS[extra].capitalize()} requires optional dependencies ({err.name or err}). "
            f'Install them with: pip install "falcondale[{extra}]"'
        ) from err
