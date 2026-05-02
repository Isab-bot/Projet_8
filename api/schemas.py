"""Schémas Pydantic pour la validation des entrées/sorties de l'API.
Fichier généré automatiquement par scripts/generate_input_schema.py
Source de vérité : pipeline.preprocessor.feature_names_in_ (326 features)
Tous les champs sont obligatoires (le modèle a été entraîné sans NaN).
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input du endpoint /predict — 326 features attendues par le modèle."""

    model_config = {
        "populate_by_name": True,
    }

    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    NAME_FAMILY_STATUS: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    CNT_FAM_MEMBERS: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    OBS_60_CNT_SOCIAL_CIRCLE: int
    DEF_60_CNT_SOCIAL_CIRCLE: int
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_8: int
    AMT_REQ_CREDIT_BUREAU_MON: int
    AMT_REQ_CREDIT_BUREAU_QRT: int
    AMT_REQ_CREDIT_BUREAU_YEAR: int
    DAYS_EMPLOYED_ANOM: bool
    CREDIT_INCOME_PERCENT: float
    ANNUITY_INCOME_PERCENT: float
    CREDIT_TERM: float
    NAME_HOUSING_TYPE_GRP: str
    NAME_EDUCATION_TYPE_GRP: str
    NAME_INCOME_TYPE_GRP: str
    NAME_TYPE_SUITE_GRP: str
    ORG_GROUP: str
    OCCUPATION_TYPE_GRP: str
    BUREAU_DAYS_CREDIT_mean: float
    BUREAU_DAYS_CREDIT_max: float
    BUREAU_DAYS_CREDIT_sum: float
    BUREAU_CREDIT_DAY_OVERDUE_mean: float
    BUREAU_CREDIT_DAY_OVERDUE_max: float
    BUREAU_CREDIT_DAY_OVERDUE_sum: float
    BUREAU_DAYS_CREDIT_ENDDATE_mean: float
    BUREAU_DAYS_CREDIT_ENDDATE_max: float
    BUREAU_DAYS_CREDIT_ENDDATE_sum: float
    BUREAU_DAYS_ENDDATE_FACT_mean: float
    BUREAU_DAYS_ENDDATE_FACT_max: float
    BUREAU_DAYS_ENDDATE_FACT_sum: float
    BUREAU_AMT_CREDIT_MAX_OVERDUE_mean: float
    BUREAU_AMT_CREDIT_MAX_OVERDUE_max: float
    BUREAU_AMT_CREDIT_MAX_OVERDUE_sum: float
    BUREAU_CNT_CREDIT_PROLONG_mean: float
    BUREAU_CNT_CREDIT_PROLONG_max: float
    BUREAU_CNT_CREDIT_PROLONG_sum: float
    BUREAU_AMT_CREDIT_SUM_mean: float
    BUREAU_AMT_CREDIT_SUM_max: float
    BUREAU_AMT_CREDIT_SUM_sum: float
    BUREAU_AMT_CREDIT_SUM_DEBT_mean: float
    BUREAU_AMT_CREDIT_SUM_DEBT_max: float
    BUREAU_AMT_CREDIT_SUM_DEBT_sum: float
    BUREAU_AMT_CREDIT_SUM_LIMIT_mean: float
    BUREAU_AMT_CREDIT_SUM_LIMIT_max: float
    BUREAU_AMT_CREDIT_SUM_LIMIT_sum: float
    BUREAU_AMT_CREDIT_SUM_OVERDUE_mean: float
    BUREAU_AMT_CREDIT_SUM_OVERDUE_max: float
    BUREAU_AMT_CREDIT_SUM_OVERDUE_sum: float
    BUREAU_DAYS_CREDIT_UPDATE_mean: float
    BUREAU_DAYS_CREDIT_UPDATE_max: float
    BUREAU_DAYS_CREDIT_UPDATE_sum: float
    BUREAU_HAD_MAX_OVERDUE_mean: float
    BUREAU_HAD_MAX_OVERDUE_max: float
    BUREAU_HAD_MAX_OVERDUE_sum: float
    BUREAU_HAD_ENDDATE_FACT_mean: float
    BUREAU_HAD_ENDDATE_FACT_max: float
    BUREAU_HAD_ENDDATE_FACT_sum: float
    BUREAU_BB_MONTHS_MIN_mean: float
    BUREAU_BB_MONTHS_MIN_max: float
    BUREAU_BB_MONTHS_MIN_sum: float
    BUREAU_BB_MONTHS_MAX_mean: float
    BUREAU_BB_MONTHS_MAX_max: float
    BUREAU_BB_MONTHS_MAX_sum: float
    BUREAU_BB_MONTHS_COUNT_mean: float
    BUREAU_BB_MONTHS_COUNT_max: float
    BUREAU_BB_MONTHS_COUNT_sum: float
    BUREAU_BB_STATUS_NO_DPD_COUNT_mean: float
    BUREAU_BB_STATUS_NO_DPD_COUNT_max: float
    BUREAU_BB_STATUS_NO_DPD_COUNT_sum: float
    BUREAU_BB_STATUS_DPD_COUNT_mean: float
    BUREAU_BB_STATUS_DPD_COUNT_max: float
    BUREAU_BB_STATUS_DPD_COUNT_sum: float
    BUREAU_BB_STATUS_UNKNOWN_COUNT_mean: float
    BUREAU_BB_STATUS_UNKNOWN_COUNT_max: float
    BUREAU_BB_STATUS_UNKNOWN_COUNT_sum: float
    BUREAU_BB_MONTHS_HISTORY_mean: float
    BUREAU_BB_MONTHS_HISTORY_max: float
    BUREAU_BB_MONTHS_HISTORY_sum: float
    BUREAU_BB_NO_DPD_RATIO_mean: float
    BUREAU_BB_NO_DPD_RATIO_max: float
    BUREAU_BB_NO_DPD_RATIO_sum: float
    BUREAU_BB_DPD_RATIO_mean: float
    BUREAU_BB_DPD_RATIO_max: float
    BUREAU_BB_DPD_RATIO_sum: float
    BUREAU_BB_UNKNOWN_RATIO_mean: float
    BUREAU_BB_UNKNOWN_RATIO_max: float
    BUREAU_BB_UNKNOWN_RATIO_sum: float
    BUREAU_HAS_CLOSED_CREDIT_mean: float
    BUREAU_HAS_CLOSED_CREDIT_max: float
    BUREAU_HAS_CLOSED_CREDIT_sum: float
    BUREAU_SK_ID_BUREAU_count: float
    BUREAU_CREDIT_ACTIVE_lambda: float = Field(..., alias="BUREAU_CREDIT_ACTIVE_<lambda>")
    BUREAU_CREDIT_TYPE_nunique: float
    POS_MONTHS_BALANCE_min_mean: float
    POS_MONTHS_BALANCE_min_max: float
    POS_MONTHS_BALANCE_min_sum: float
    POS_MONTHS_BALANCE_max_mean: float
    POS_MONTHS_BALANCE_max_max: float
    POS_MONTHS_BALANCE_max_sum: float
    POS_MONTHS_BALANCE_size_mean: float
    POS_MONTHS_BALANCE_size_max: float
    POS_MONTHS_BALANCE_size_sum: float
    POS_CNT_INSTALMENT_mean_mean: float
    POS_CNT_INSTALMENT_mean_max: float
    POS_CNT_INSTALMENT_mean_sum: float
    POS_CNT_INSTALMENT_max_mean: float
    POS_CNT_INSTALMENT_max_max: float
    POS_CNT_INSTALMENT_max_sum: float
    POS_CNT_INSTALMENT_min_mean: float
    POS_CNT_INSTALMENT_min_max: float
    POS_CNT_INSTALMENT_min_sum: float
    POS_CNT_INSTALMENT_FUTURE_mean_mean: float
    POS_CNT_INSTALMENT_FUTURE_mean_max: float
    POS_CNT_INSTALMENT_FUTURE_mean_sum: float
    POS_CNT_INSTALMENT_FUTURE_max_mean: float
    POS_CNT_INSTALMENT_FUTURE_max_max: float
    POS_CNT_INSTALMENT_FUTURE_max_sum: float
    POS_CNT_INSTALMENT_FUTURE_min_mean: float
    POS_CNT_INSTALMENT_FUTURE_min_max: float
    POS_CNT_INSTALMENT_FUTURE_min_sum: float
    POS_SK_DPD_mean_mean: float
    POS_SK_DPD_mean_max: float
    POS_SK_DPD_mean_sum: float
    POS_SK_DPD_max_mean: float
    POS_SK_DPD_max_max: float
    POS_SK_DPD_max_sum: float
    POS_SK_DPD_sum_mean: float
    POS_SK_DPD_sum_max: float
    POS_SK_DPD_sum_sum: float
    POS_SK_DPD_DEF_mean_mean: float
    POS_SK_DPD_DEF_mean_max: float
    POS_SK_DPD_DEF_mean_sum: float
    POS_SK_DPD_DEF_max_mean: float
    POS_SK_DPD_DEF_max_max: float
    POS_SK_DPD_DEF_max_sum: float
    POS_SK_DPD_DEF_sum_mean: float
    POS_SK_DPD_DEF_sum_max: float
    POS_SK_DPD_DEF_sum_sum: float
    POS_NAME_CONTRACT_STATUS_lambda_0_mean: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_0>_mean")
    POS_NAME_CONTRACT_STATUS_lambda_0_max: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_0>_max")
    POS_NAME_CONTRACT_STATUS_lambda_0_sum: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_0>_sum")
    POS_NAME_CONTRACT_STATUS_lambda_1_mean: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_1>_mean")
    POS_NAME_CONTRACT_STATUS_lambda_1_max: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_1>_max")
    POS_NAME_CONTRACT_STATUS_lambda_1_sum: float = Field(..., alias="POS_NAME_CONTRACT_STATUS_<lambda_1>_sum")
    POS_MONTHS_HISTORY_mean: float
    POS_MONTHS_HISTORY_max: float
    POS_MONTHS_HISTORY_sum: float
    POS_DPD_RATIO_mean: float
    POS_DPD_RATIO_max: float
    POS_DPD_RATIO_sum: float
    POS_NB_CREDITS: float
    CC_MONTHS_BALANCE_min_mean: float
    CC_MONTHS_BALANCE_min_max: float
    CC_MONTHS_BALANCE_min_sum: float
    CC_MONTHS_BALANCE_max_mean: float
    CC_MONTHS_BALANCE_max_max: float
    CC_MONTHS_BALANCE_max_sum: float
    CC_MONTHS_BALANCE_size_mean: float
    CC_MONTHS_BALANCE_size_max: float
    CC_MONTHS_BALANCE_size_sum: float
    CC_AMT_BALANCE_mean_mean: float
    CC_AMT_BALANCE_mean_max: float
    CC_AMT_BALANCE_mean_sum: float
    CC_AMT_BALANCE_max_mean: float
    CC_AMT_BALANCE_max_max: float
    CC_AMT_BALANCE_max_sum: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_mean_mean: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_mean_max: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_mean_sum: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_max_mean: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_max_max: float
    CC_AMT_CREDIT_LIMIT_ACTUAL_max_sum: float
    CC_SK_DPD_mean_mean: float
    CC_SK_DPD_mean_max: float
    CC_SK_DPD_mean_sum: float
    CC_SK_DPD_max_mean: float
    CC_SK_DPD_max_max: float
    CC_SK_DPD_max_sum: float
    CC_SK_DPD_sum_mean: float
    CC_SK_DPD_sum_max: float
    CC_SK_DPD_sum_sum: float
    CC_SK_DPD_DEF_mean_mean: float
    CC_SK_DPD_DEF_mean_max: float
    CC_SK_DPD_DEF_mean_sum: float
    CC_SK_DPD_DEF_max_mean: float
    CC_SK_DPD_DEF_max_max: float
    CC_SK_DPD_DEF_max_sum: float
    CC_SK_DPD_DEF_sum_mean: float
    CC_SK_DPD_DEF_sum_max: float
    CC_SK_DPD_DEF_sum_sum: float
    CC_CC_CREDIT_UTILIZATION_mean_mean: float
    CC_CC_CREDIT_UTILIZATION_mean_max: float
    CC_CC_CREDIT_UTILIZATION_mean_sum: float
    CC_CC_CREDIT_UTILIZATION_max_mean: float
    CC_CC_CREDIT_UTILIZATION_max_max: float
    CC_CC_CREDIT_UTILIZATION_max_sum: float
    CC_CC_PAYMENT_RATIO_mean_mean: float
    CC_CC_PAYMENT_RATIO_mean_max: float
    CC_CC_PAYMENT_RATIO_mean_sum: float
    CC_CC_PAYMENT_RATIO_min_mean: float
    CC_CC_PAYMENT_RATIO_min_max: float
    CC_CC_PAYMENT_RATIO_min_sum: float
    CC_NB_CARDS: float
    INST_NUM_INSTALMENT_NUMBER_max_mean: float
    INST_NUM_INSTALMENT_NUMBER_max_max: float
    INST_NUM_INSTALMENT_NUMBER_max_sum: float
    INST_NUM_INSTALMENT_NUMBER_count_mean: float
    INST_NUM_INSTALMENT_NUMBER_count_max: float
    INST_NUM_INSTALMENT_NUMBER_count_sum: float
    INST_AMT_INSTALMENT_sum_mean: float
    INST_AMT_INSTALMENT_sum_max: float
    INST_AMT_INSTALMENT_sum_sum: float
    INST_AMT_PAYMENT_sum_mean: float
    INST_AMT_PAYMENT_sum_max: float
    INST_AMT_PAYMENT_sum_sum: float
    INST_INST_PAYMENT_DIFF_mean_mean: float
    INST_INST_PAYMENT_DIFF_mean_max: float
    INST_INST_PAYMENT_DIFF_mean_sum: float
    INST_INST_PAYMENT_DIFF_min_mean: float
    INST_INST_PAYMENT_DIFF_min_max: float
    INST_INST_PAYMENT_DIFF_min_sum: float
    INST_INST_PAYMENT_DIFF_max_mean: float
    INST_INST_PAYMENT_DIFF_max_max: float
    INST_INST_PAYMENT_DIFF_max_sum: float
    INST_INST_DAYS_DELAY_mean_mean: float
    INST_INST_DAYS_DELAY_mean_max: float
    INST_INST_DAYS_DELAY_mean_sum: float
    INST_INST_DAYS_DELAY_max_mean: float
    INST_INST_DAYS_DELAY_max_max: float
    INST_INST_DAYS_DELAY_max_sum: float
    INST_INST_IS_LATE_sum_mean: float
    INST_INST_IS_LATE_sum_max: float
    INST_INST_IS_LATE_sum_sum: float
    INST_INST_IS_LATE_mean_mean: float
    INST_INST_IS_LATE_mean_max: float
    INST_INST_IS_LATE_mean_sum: float
    INST_NB_CREDITS: float
    PREV_AMT_ANNUITY_mean: float
    PREV_AMT_ANNUITY_max: float
    PREV_AMT_ANNUITY_sum: float
    PREV_AMT_APPLICATION_mean: float
    PREV_AMT_APPLICATION_max: float
    PREV_AMT_APPLICATION_sum: float
    PREV_AMT_CREDIT_mean: float
    PREV_AMT_CREDIT_max: float
    PREV_AMT_CREDIT_sum: float
    PREV_AMT_DOWN_PAYMENT_mean: float
    PREV_AMT_DOWN_PAYMENT_max: float
    PREV_AMT_DOWN_PAYMENT_sum: float
    PREV_AMT_GOODS_PRICE_mean: float
    PREV_AMT_GOODS_PRICE_max: float
    PREV_AMT_GOODS_PRICE_sum: float
    PREV_HOUR_APPR_PROCESS_START_mean: float
    PREV_HOUR_APPR_PROCESS_START_max: float
    PREV_HOUR_APPR_PROCESS_START_sum: float
    PREV_NFLAG_LAST_APPL_IN_DAY_mean: float
    PREV_NFLAG_LAST_APPL_IN_DAY_max: float
    PREV_NFLAG_LAST_APPL_IN_DAY_sum: float
    PREV_RATE_DOWN_PAYMENT_mean: float
    PREV_RATE_DOWN_PAYMENT_max: float
    PREV_RATE_DOWN_PAYMENT_sum: float
    PREV_DAYS_DECISION_mean: float
    PREV_DAYS_DECISION_max: float
    PREV_DAYS_DECISION_sum: float
    PREV_SELLERPLACE_AREA_mean: float
    PREV_SELLERPLACE_AREA_max: float
    PREV_SELLERPLACE_AREA_sum: float
    PREV_CNT_PAYMENT_mean: float
    PREV_CNT_PAYMENT_max: float
    PREV_CNT_PAYMENT_sum: float
    PREV_DAYS_FIRST_DRAWING_mean: float
    PREV_DAYS_FIRST_DRAWING_max: float
    PREV_DAYS_FIRST_DRAWING_sum: float
    PREV_DAYS_FIRST_DUE_mean: float
    PREV_DAYS_FIRST_DUE_max: float
    PREV_DAYS_FIRST_DUE_sum: float
    PREV_DAYS_LAST_DUE_1ST_VERSION_mean: float
    PREV_DAYS_LAST_DUE_1ST_VERSION_max: float
    PREV_DAYS_LAST_DUE_1ST_VERSION_sum: float
    PREV_DAYS_LAST_DUE_mean: float
    PREV_DAYS_LAST_DUE_max: float
    PREV_DAYS_LAST_DUE_sum: float
    PREV_DAYS_TERMINATION_mean: float
    PREV_DAYS_TERMINATION_max: float
    PREV_DAYS_TERMINATION_sum: float
    PREV_NFLAG_INSURED_ON_APPROVAL_mean: float
    PREV_NFLAG_INSURED_ON_APPROVAL_max: float
    PREV_NFLAG_INSURED_ON_APPROVAL_sum: float
    PREV_PREV_CREDIT_DIFF_mean: float
    PREV_PREV_CREDIT_DIFF_max: float
    PREV_PREV_CREDIT_DIFF_sum: float
    PREV_PREV_CREDIT_RATIO_mean: float
    PREV_PREV_CREDIT_RATIO_max: float
    PREV_PREV_CREDIT_RATIO_sum: float
    PREV_SK_ID_PREV_count: float
    PREV_NAME_CONTRACT_STATUS_lambda_0: float = Field(..., alias="PREV_NAME_CONTRACT_STATUS_<lambda_0>")
    PREV_NAME_CONTRACT_STATUS_lambda_1: float = Field(..., alias="PREV_NAME_CONTRACT_STATUS_<lambda_1>")
    PREV_NAME_CONTRACT_STATUS_lambda_2: float = Field(..., alias="PREV_NAME_CONTRACT_STATUS_<lambda_2>")
    PREV_APPROVAL_RATIO: float


class PredictionOutput(BaseModel):
    """Sortie du endpoint /predict."""

    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probabilité prédite de défaut de crédit (classe 1)."
    )
    decision: int = Field(
        ..., ge=0, le=1,
        description="Décision binaire : 0 = crédit accordé, 1 = défaut prédit."
    )
    threshold: float = Field(
        ...,
        description="Seuil de décision utilisé (0.3338, optimal F3 du Projet 6)."
    )


class HealthResponse(BaseModel):
    """Sortie du endpoint /health."""

    status: str
    model_loaded: bool
    api_version: str
