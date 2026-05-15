"""Tests unitaires de api/feature_naming.py."""
from api.feature_naming import to_pandas_feature_name, to_sql_column_name


class TestToSqlColumnName:
    """Conversion pandas → SQL : suppression des chevrons et collapse des underscores."""

    def test_simple_lambda_pattern(self) -> None:
        """Le motif <lambda> est remplacé par lambda."""
        assert (
            to_sql_column_name("BUREAU_CREDIT_ACTIVE_<lambda>")
            == "BUREAU_CREDIT_ACTIVE_lambda"
        )

    def test_lambda_with_index(self) -> None:
        """Le motif <lambda_0> conserve l'index numérique."""
        assert (
            to_sql_column_name("POS_NAME_CONTRACT_STATUS_<lambda_0>_mean")
            == "POS_NAME_CONTRACT_STATUS_lambda_0_mean"
        )

    def test_lambda_at_end(self) -> None:
        """Le motif <lambda_2> en fin de nom est correctement traité."""
        assert (
            to_sql_column_name("PREV_NAME_CONTRACT_STATUS_<lambda_2>")
            == "PREV_NAME_CONTRACT_STATUS_lambda_2"
        )

    def test_no_special_chars_unchanged(self) -> None:
        """Un nom déjà valide est retourné tel quel (idempotence partielle)."""
        assert to_sql_column_name("EXT_SOURCE_1") == "EXT_SOURCE_1"

    def test_idempotent_on_already_converted(self) -> None:
        """Appliquer la fonction deux fois donne le même résultat."""
        once = to_sql_column_name("BUREAU_CREDIT_ACTIVE_<lambda>")
        twice = to_sql_column_name(once)
        assert once == twice

    def test_double_underscore_collapsed(self) -> None:
        """Les doubles underscores créés par la suppression sont collapsés."""
        # Cas où la suppression du motif générerait __ : <X>_ encadré par _
        assert to_sql_column_name("A_<x>_B") == "A_x_B"
        # Cas explicite de double underscore en entrée
        assert to_sql_column_name("A__B") == "A_B"


class TestToPandasFeatureName:
    """Reverse mapping SQL → pandas via lookup dans la liste d'origine."""

    def test_reverse_mapping_found(self) -> None:
        """Si le nom SQL correspond à un nom d'origine, on retrouve l'original."""
        originals = ["EXT_SOURCE_1", "BUREAU_CREDIT_ACTIVE_<lambda>"]
        assert (
            to_pandas_feature_name("BUREAU_CREDIT_ACTIVE_lambda", originals)
            == "BUREAU_CREDIT_ACTIVE_<lambda>"
        )

    def test_reverse_mapping_not_found_returns_input(self) -> None:
        """Si aucune correspondance, on renvoie l'entrée telle quelle."""
        originals = ["EXT_SOURCE_1", "EXT_SOURCE_2"]
        assert (
            to_pandas_feature_name("UNKNOWN_COLUMN", originals) == "UNKNOWN_COLUMN"
        )

    def test_reverse_mapping_passthrough_for_simple_names(self) -> None:
        """Un nom sans transformation est retrouvé identiquement."""
        originals = ["EXT_SOURCE_1", "EXT_SOURCE_2"]
        assert to_pandas_feature_name("EXT_SOURCE_1", originals) == "EXT_SOURCE_1"
