import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for preprocessing numeric and categorical columns.
    """
    logger.info("Building preprocessor pipeline components.")
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor

def create_full_pipeline(
    preprocessor: ColumnTransformer, 
    model, 
    use_smote: bool = True,
    random_state: int = RANDOM_STATE
) -> Pipeline:
    """
    Combines preprocessor, optional SMOTE, and classifier into a single pipeline.
    """
    if use_smote:
        logger.info(f"Building Imbalanced-Learn Pipeline with SMOTE (random_state={random_state}) and model {model.__class__.__name__}")
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", model)
        ])
    else:
        logger.info(f"Building Standard Scikit-Learn Pipeline with model {model.__class__.__name__}")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
    return pipeline
