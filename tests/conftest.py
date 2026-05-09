"""Fixtures partagées pour les tests pytest.

Architecture :
- DB de test = SQLite fichier temporaire (session-scoped)
- Tables créées via Base.metadata.create_all (pas Alembic)
- Isolation entre tests via DELETE FROM (function-scoped)
- TestClient FastAPI avec dépendance get_db overridée
"""
from __future__ import annotations

import os
import json

# IMPORTANT : doit être positionné AVANT l'import de api.main
os.environ["SKIP_ALEMBIC_ON_STARTUP"] = "1"

from collections.abc import Generator
from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.database import Base, get_db
from api.main import app
from api.models import Prediction


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Chemin du fichier SQLite temporaire, partagé pour toute la session."""
    db_dir = tmp_path_factory.mktemp("test_db")
    return db_dir / "test_predictions.db"


@pytest.fixture(scope="session")
def test_engine(test_db_path: Path) -> Generator[Engine, None, None]:
    """Engine SQLAlchemy de test, tables créées une fois pour la session."""
    engine = create_engine(
        f"sqlite:///{test_db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Session par test, avec nettoyage de la table après chaque test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Nettoyage : on vide la table predictions entre chaque test
        with test_engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {Prediction.__tablename__}"))
            conn.commit()


@pytest.fixture(scope="function")
def client(test_engine: Engine) -> Generator[TestClient, None, None]:
    """TestClient FastAPI avec get_db pointant sur la DB de test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

    # Nettoyage entre tests
    with test_engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {Prediction.__tablename__}"))
        conn.commit()

@pytest.fixture(scope="session")
def sample_payload_dict() -> dict:
    """Payload de test (326 features avec noms d'origine, alias <lambda>).

    Source : tests/fixtures/sample_request.json (extrait de reference_data.parquet,
    ligne 0). Versionné dans le repo pour que les tests tournent en CI.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "sample_request.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_prediction_input(sample_payload_dict: dict):
    """PredictionInput Pydantic construit depuis le payload de test.

    Validé par Pydantic (model_validate), donc équivalent à un POST /predict
    avec un body JSON valide.
    """
    from api.schemas import PredictionInput
    return PredictionInput.model_validate(sample_payload_dict)
    