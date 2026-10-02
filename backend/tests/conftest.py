import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from selene.api.app import create_app

    return TestClient(create_app())
