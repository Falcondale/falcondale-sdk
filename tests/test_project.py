import pytest

from falcondale import Model, Project


def test_list_options(capsys):
    options = Project.list_options()
    assert options["feature_selection"] == ["exact", "qaoa", "sa", "sb"]
    assert options["evaluate"] == ["qnn", "qsvc", "svc"]
    assert "Falcondale" in capsys.readouterr().out


def test_feature_selection_updates_features(project):
    features = project.feature_selection(max_cols=3, seed=0)
    assert list(project.show_features().columns) == features


def test_feature_selection_beyond_available(project):
    with pytest.raises(ValueError):
        project.feature_selection(max_cols=100)


def test_unknown_cluster(project):
    with pytest.raises(ValueError):
        project.cluster("kmeans")


def test_evaluate_save_and_load(project, breast_cancer, tmp_path):
    project.feature_selection(max_cols=3, seed=0)
    model = project.evaluate("svc")

    assert isinstance(model, Model)
    assert "auc" in model.list_metrics()
    with pytest.raises(KeyError):
        model.metric("unknown")

    path = tmp_path / "model.pkl"
    project.save_model(model, path)
    restored = Project.load_model(path)
    assert (restored.predict(breast_cancer) == model.predict(breast_cancer)).all()
    assert restored.predict_proba(breast_cancer).shape == (len(breast_cancer),)
