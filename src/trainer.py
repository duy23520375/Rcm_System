import xgboost as xgb
import joblib
import polars as pl

def train_model(train_df, feature_cols):
    train_pd = train_df.collect().with_columns([
        pl.col(feature_cols).cast(pl.Float64).fill_null(0)
    ]).to_pandas()

    X_train = train_pd[feature_cols]
    y_train = train_pd["Y"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'nthread': -1
    }
    
    print("Training XGBoost...")
    model = xgb.train(params, dtrain, num_boost_round=500)
    return model