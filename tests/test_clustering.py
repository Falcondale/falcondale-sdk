import pytest

pytest.importorskip("apqc")


@pytest.mark.slow
def test_pqc_can_be_called_repeatedly(project):
    """Calling the clustering twice used to fail (GitHub issue #26)."""
    project.preprocess(reduced_dimension=2)
    for _ in range(2):
        labels, probabilities = project.cluster("pqc", sigma=0.3)
        assert len(labels) == len(probabilities) == len(project.show_features())
