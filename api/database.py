"""Configuration du moteur de base de données, fabrique de sessions et base ORM.

Ce module configure SQLAlchemy 2.0 avec une stratégie à deux backends :
- PostgreSQL (via psycopg v3) lorsque DATABASE_URL pointe vers une instance Postgres.
- SQLite en fallback pour la CI et Hugging Face Spaces.

Le choix est transparent pour le code applicatif ; seule la variable
DATABASE_URL change selon l'environnement.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import DATABASE_URL

# --- Moteur (engine) ---
# Spécificité SQLite : par défaut SQLite interdit le partage d'une connexion
# entre threads. Or FastAPI exécute les endpoints synchrones dans un pool de
# threads, donc il faut autoriser l'usage cross-thread. PostgreSQL n'a pas
# cette contrainte.
_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,  # passer à True en local pour voir les requêtes SQL
    future=True,  # comportement SQLAlchemy 2.0 explicite (déjà le défaut)
)

# --- Fabrique de sessions ---
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# --- Base ORM déclarative ---
class Base(DeclarativeBase):
    """Classe parente de tous les modèles ORM (style SQLAlchemy 2.0)."""
    pass


# --- Dépendance FastAPI ---
def get_db() -> Generator[Session, None, None]:
    """Fournit une session DB par requête, fermée automatiquement à la fin.

    Utilisation dans un endpoint :
        def mon_endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
