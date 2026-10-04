import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from scripts.make_samples import build_all


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("samples"))


@pytest.fixture
def make_client(tmp_path):
    def _make(**kwargs):
        return TestClient(create_app(data_dir=tmp_path / "data", **kwargs))
    return _make


@pytest.fixture
def client(make_client):
    return make_client()
