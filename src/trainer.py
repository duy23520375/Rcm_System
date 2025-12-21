import xgboost as xgb
import joblib
import polars as pl

def train_model(train_df, feature_cols):
    print("⏳ Collecting and processing training data...")
    
    # 1. Chuyển đổi an toàn: 
    # Ép kiểu features về Float32 và đảm bảo nhãn Y là số nguyên cho XGBoost
    train_pd = (
        train_df
        .select(feature_cols + ["Y"]) # Chỉ lấy các cột cần thiết để tiết kiệm RAM
        .with_columns([
            pl.col(feature_cols).cast(pl.Float32).fill_null(0),
            pl.col("Y").cast(pl.Int8) # Nhãn 0/1 chỉ cần Int8
        ])
        .collect()
        .to_pandas()
    )

    X_train = train_pd[feature_cols]
    y_train = train_pd["Y"]

    # 2. Khởi tạo DMatrix (định dạng tối ưu của XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # 3. Cấu hình tham số
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  
        'eta': 0.05,               # Tăng nhẹ tốc độ học nếu bạn thấy 0.03 quá chậm cho 800 rounds
        'max_depth': 8,            # 10 hơi sâu dễ bị Overfit, 8 là điểm cân bằng tốt cho Precision
        'subsample': 0.8,          
        'colsample_bytree': 0.8,   
        'tree_method': 'hist',     # Bắt buộc cho dữ liệu lớn (>1M dòng)
        'device': 'cpu',           # Đảm bảo chạy ổn định trên CPU
        'nthread': -1              
    }
    
    print(f"🚀 Training XGBoost with {X_train.shape[0]:,} samples...")
    # Huấn luyện model
    model = xgb.train(params, dtrain, num_boost_round=500) # 500 rounds thường là đủ cho 3 features
    return model