"""
Récupère le model_id et le path .pkl du vrai champion P6 en lisant
directement la base SQLite MLflow (sans dépendre de MLflow lui-même).

Run cible : a3ff1e12347c4bfc9b484ac36916eb14
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\MLflow_tracking\credit_risk\mlflow.db")
TRUE_CHAMPION_RUN_ID = "a3ff1e12347c4bfc9b484ac36916eb14"
MLRUNS_ROOT = Path(r"C:\Users\renar\Documents\Alternance\Projet_6\MLOps_1\mlruns")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Base MLflow introuvable : {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- 1. Lister toutes les tables (utile pour comprendre la structure) ---
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r["name"] for r in cur.fetchall()]
    print("Tables disponibles :", tables)
    print()

    # --- 2. Infos du run ---
    print("=" * 70)
    print("INFOS DU RUN CHAMPION")
    print("=" * 70)
    cur.execute(
        "SELECT run_uuid, name, status, experiment_id, artifact_uri "
        "FROM runs WHERE run_uuid = ?",
        (TRUE_CHAMPION_RUN_ID,),
    )
    run_row = cur.fetchone()
    if run_row is None:
        raise RuntimeError(f"Run {TRUE_CHAMPION_RUN_ID} introuvable.")
    for k in run_row.keys():
        print(f"  {k:14s} = {run_row[k]}")

    # --- 3. Métriques pertinentes ---
    print("\n--- Métriques (test_f3*, threshold) ---")
    cur.execute(
        "SELECT key, value FROM metrics WHERE run_uuid = ? "
        "AND (key LIKE '%f3%' OR key LIKE '%threshold%' OR key LIKE '%test%')",
        (TRUE_CHAMPION_RUN_ID,),
    )
    for r in cur.fetchall():
        print(f"  {r['key']:30s} = {r['value']}")

    # --- 4. Params pertinents ---
    print("\n--- Params (threshold, model_type) ---")
    cur.execute(
        "SELECT key, value FROM params WHERE run_uuid = ? "
        "AND (key LIKE '%threshold%' OR key LIKE '%model_type%' OR key LIKE '%f3%')",
        (TRUE_CHAMPION_RUN_ID,),
    )
    for r in cur.fetchall():
        print(f"  {r['key']:30s} = {r['value']}")

    # --- 5. Tags ---
    print("\n--- Tags pertinents ---")
    cur.execute(
        "SELECT key, value FROM tags WHERE run_uuid = ?",
        (TRUE_CHAMPION_RUN_ID,),
    )
    for r in cur.fetchall():
        print(f"  {r['key']:35s} = {r['value']}")

    # --- 6. Logged models (MLflow ≥ 2.9) ---
    print("\n" + "=" * 70)
    print("LOGGED MODELS LIÉS À CE RUN")
    print("=" * 70)

    if "logged_models" in tables:
        # Inspecter la structure
        cur.execute("PRAGMA table_info(logged_models)")
        cols = [r["name"] for r in cur.fetchall()]
        print(f"Colonnes logged_models : {cols}\n")

        # Chercher par source_run_id
        if "source_run_id" in cols:
            cur.execute(
                "SELECT * FROM logged_models WHERE source_run_id = ?",
                (TRUE_CHAMPION_RUN_ID,),
            )
            rows = cur.fetchall()
            if not rows:
                print("(Aucune ligne logged_models pour ce run)")
            for r in rows:
                print("--- Logged model ---")
                for k in r.keys():
                    print(f"  {k:25s} = {r[k]}")
                # Construction du path .pkl
                model_id = r["model_id"] if "model_id" in r.keys() else None
                exp_id = run_row["experiment_id"]
                if model_id:
                    pkl_path = (
                        MLRUNS_ROOT / str(exp_id) / "models" / model_id
                        / "artifacts" / "model.pkl"
                    )
                    print(f"\n  >>> Path .pkl reconstruit : {pkl_path}")
                    print(f"  >>> Existe ? : {pkl_path.exists()}")
                    if pkl_path.exists():
                        size_mb = pkl_path.stat().st_size / (1024 * 1024)
                        print(f"  >>> Taille    : {size_mb:.2f} Mo")
    else:
        print("(table 'logged_models' absente — MLflow plus ancien)")

    conn.close()


if __name__ == "__main__":
    main()
