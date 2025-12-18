# ======================================================================
# 5. INFERENCE (CHẠY TRỰC TIẾP TRONG NOTEBOOK)
# ======================================================================
import xgboost as xgb
import polars as pl
import numpy as np

def run_inference_local(model, infer_features_df, lf_items, lf_purchases, top_k=10, batch_size=5000):
    # 1. Tạo Candidate Pool (Top 200 món phổ biến)
    print("Step 1: Creating Candidate Pool...")
    popular_items_list = (
        lf_purchases.group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(200)
        .select("item_id")
        .collect()
        .get_column("item_id")
        .to_list()
    )
    popular_items_lf = pl.DataFrame({"X_0": popular_items_list}).lazy()

    # 2. Lấy danh sách khách hàng
    all_users = (
        infer_features_df
        .select("X_-1")
        .unique()
        .collect()
        .get_column("X_-1")
        .to_list()
    )
    print(f"Step 2: Processing {len(all_users):,} users in batches of {batch_size}...")

    all_recommendations = []

    # 3. Chạy theo Batch
    for i in range(0, len(all_users), batch_size):
        batch_u = all_users[i : i + batch_size]
        batch_u_lf = pl.DataFrame({"X_-1": batch_u}).lazy()

        # Tạo candidates (Lazy x Lazy)
        candidates_lazy = batch_u_lf.join(popular_items_lf, how="cross")

        # Join với feature (Lazy x Lazy) -> Sửa lỗi TypeError ở đây
        batch_infer_df_lazy = (
            candidates_lazy
            .join(
                infer_features_df.filter(pl.col("X_-1").is_in(batch_u)),
                on=["X_-1", "X_0"],
                how="left"
            )
            .with_columns(
                pl.all().exclude(["X_-1", "X_0"]).fill_null(0)
            )
        )

        # Chuyển sang DataFrame để nạp vào XGBoost
        batch_data = batch_infer_df_lazy.collect()
        
        # Predict
        X_infer = batch_data.select(["X_1", "X_2", "X_3"]).to_pandas()
        dtest = xgb.DMatrix(X_infer)
        
        batch_result = batch_data.with_columns(
            pl.Series(model.predict(dtest)).alias("prob")
        )

        # Lọc món đã mua
        purchased_items = (
            lf_purchases.filter(pl.col("customer_id").is_in(batch_u))
            .select([
                pl.col("customer_id").alias("X_-1"), 
                pl.col("item_id").alias("X_0")
            ])
            .collect()
        )
        
        # Lấy Top K
        top_recs = (
            batch_result.join(
                purchased_items,
                on=["X_-1", "X_0"],
                how="anti"
            )
            .sort(["X_-1", "prob"], descending=[False, True])
            .group_by("X_-1")
            .head(top_k)
            .select(["X_-1", "X_0"])
        )
        
        all_recommendations.append(top_recs)
        
        if (i // batch_size) % 10 == 0:
            print(f"   Processed {i + len(batch_u):,} / {len(all_users):,} users...")

    return pl.concat(all_recommendations)

# Thực thi hàm vừa định nghĩa
recommendations_warm = run_inference_local(
    model=model_xgb,
    infer_features_df=infer_features_df,
    lf_items=lf_items,
    lf_purchases=full_history,
    top_k=10,
    batch_size=10000 
)

print(f"✅ Đã có kết quả dự đoán cho {recommendations_warm.select('X_-1').n_unique()} users.")
display(recommendations_warm.head(10))