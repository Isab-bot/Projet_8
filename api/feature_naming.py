"""Conversion des noms de features entre Pandas/Pydantic et SQL.

Certaines features héritées du Projet 6 contiennent des caractères spéciaux
qui ne sont pas acceptés comme noms de colonnes SQL (notamment "<" et ">"
issus de fonctions lambda dans le feature engineering pandas).

Ce module centralise la règle de conversion pour garantir une cohérence
entre le générateur de modèle ORM, le service de logging, et toute lecture
ultérieure (monitoring Evidently, requêtes ad-hoc).

Règle appliquée :
    Tout motif <mot> est remplacé par mot (sans les chevrons).
    Les doubles underscores éventuellement créés sont collapsés en simple.

Exemples :
    BUREAU_CREDIT_ACTIVE_<lambda>            -> BUREAU_CREDIT_ACTIVE_lambda
    POS_NAME_CONTRACT_STATUS_<lambda_0>_mean -> POS_NAME_CONTRACT_STATUS_lambda_0_mean
    PREV_NAME_CONTRACT_STATUS_<lambda_2>     -> PREV_NAME_CONTRACT_STATUS_lambda_2

Cette règle est cohérente avec les noms Python utilisés dans api/schemas.py
(alias Pydantic), garantissant qu'un PredictionInput peut être inséré
directement dans la table predictions sans renommage intermédiaire.
"""
import re

# Pattern : un mot (lettres/chiffres/_) entre chevrons
_LAMBDA_PATTERN = re.compile(r"<(\w+)>")
_DOUBLE_UNDERSCORE = re.compile(r"__+")


def to_sql_column_name(feature_name: str) -> str:
    """Transforme un nom de feature pandas en nom de colonne SQL valide.

    >>> to_sql_column_name("BUREAU_CREDIT_ACTIVE_<lambda>")
    'BUREAU_CREDIT_ACTIVE_lambda'
    >>> to_sql_column_name("POS_NAME_CONTRACT_STATUS_<lambda_0>_mean")
    'POS_NAME_CONTRACT_STATUS_lambda_0_mean'
    >>> to_sql_column_name("EXT_SOURCE_1")
    'EXT_SOURCE_1'
    """
    name = _LAMBDA_PATTERN.sub(r"\1", feature_name)
    name = _DOUBLE_UNDERSCORE.sub("_", name)
    return name


def to_pandas_feature_name(sql_column_name: str, original_names: list[str]) -> str:
    """Reverse mapping : trouve le nom pandas d'origine à partir du nom SQL.

    Nécessite la liste des noms d'origine pour résoudre l'ambiguïté
    (la transformation n'est pas strictement inversible).

    Renvoie sql_column_name tel quel si aucune correspondance trouvée.
    """
    for original in original_names:
        if to_sql_column_name(original) == sql_column_name:
            return original
    return sql_column_name
