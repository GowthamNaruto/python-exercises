def Pipeline(steps):
    return 1


def ColumnTransformer(steps):
    return 1


numeric_transformers = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("encoder", OneHotEncoder()),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor)
    ]
)
