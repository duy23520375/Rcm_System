import polars as pl
import numpy as np
import time



def build_features_from_purchases(lf_purchases, lf_items, lf_users=None):
    # 1. JOIN metadata - Chú ý: item_id đều là String nên Join rất an toàn
    data_lf = lf_purchases.join(
        lf_items.select(['item_id', 'brand', 'age_group', 'category']),
        on='item_id',
        how='inner'
    )

    # 2. TÍNH TOÁN VÀ GIỮ LẠI CỘT KEY
    feature_df = (
        data_lf
        .select([
            pl.col("customer_id").alias("X_-1"), # Giữ Int32 theo Schema
            pl.col("item_id").alias("X_0"),      # Giữ String theo Schema
            pl.col("brand"),                     # String
            pl.col("category"),                  # String
            pl.col("age_group"),                 # String

            # Tính toán tần suất
            pl.len().over(["customer_id", "brand"]).alias("X_1"),
            pl.len().over(["customer_id", "age_group"]).alias("X_2"),
            pl.len().over(["customer_id", "category"]).alias("X_3"),
        ])
        .unique(subset=["X_-1", "X_0"])
        .with_columns([
            pl.col("X_1").cast(pl.Float64),
            pl.col("X_2").cast(pl.Float64),
            pl.col("X_3").cast(pl.Float64),
        ])
    )
    return feature_df

import time
import polars as pl
import numpy as np


def build_labels(
    baseHist: pl.LazyFrame,
    labelHist: pl.LazyFrame,
    items: pl.LazyFrame,
    negative_ratio: float = 1.0
) -> pl.LazyFrame:
    
    print("🚀 Building Labels with Vectorized Hard Negative Strategy...")
    
    # 1. POSITIVE SAMPLES (Mẫu dương: Những món thực sự mua)
    pos = (
        labelHist.select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.String)
        ])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )

    # 2. XÁC ĐỊNH CATEGORY CỦA TỪNG USER (Dựa trên lịch sử mua sắm)
    # Lấy top 3 category mỗi user mua nhiều nhất
    user_top_cats = (
        baseHist.join(items.select(["item_id", "category"]), on="item_id")
        .group_by(["customer_id", "category"])
        .len()
        .sort(["customer_id", "len"], descending=[False, True])
        .group_by("customer_id")
        .head(3) 
        .select([
            pl.col("customer_id").cast(pl.Int32),
            "category"
        ])
    )

    # 3. TẠO POOL MẪU ÂM THEO TRENDING CATEGORY
    # Lấy top 50 món bán chạy nhất mỗi category
    category_trending = (
        baseHist.join(items.select(["item_id", "category"]), on="item_id")
        .group_by(["category", "item_id"])
        .len()
        .sort(["category", "len"], descending=[False, True])
        .group_by("category")
        .head(50)
        .select([
            "category",
            pl.col("item_id").cast(pl.String)
        ])
    )

    # 4. GENERATE NEGATIVES (Mẫu âm khó - Hard Negatives)
    # Join User với các món hot thuộc Category họ hay mua
    neg = (
        user_top_cats.join(category_trending, on="category")
        .select(["customer_id", "item_id"])
        # Loại bỏ những món User THỰC SỰ đã mua (Positive) và món đã mua trong quá khứ (History)
        .join(pos.select(["customer_id", "item_id"]), on=["customer_id", "item_id"], how="anti")
        .join(baseHist.select(["customer_id", "item_id"]), on=["customer_id", "item_id"], how="anti")
        # Gán nhãn 0 cho mẫu âm
        .with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))
        # Giới hạn tỷ lệ mẫu âm để cân bằng dữ liệu
        .group_by("customer_id")
        .head(int(negative_ratio * 10)) 
    )

    # 5. KẾT HỢP POSITIVE VÀ NEGATIVE
    # Đảm bảo schema của cả 2 bảng khớp hoàn toàn trước khi concat
    final_labels = pl.concat([
        pos.select(["customer_id", "item_id", "label"]),
        neg.select(["customer_id", "item_id", "label"])
    ], how="vertical")
    
    return final_labels