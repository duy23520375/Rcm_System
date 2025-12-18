import polars as pl
import numpy as np
import time
def build_features_from_purchases(lf_purchases, lf_items, lf_users=None):
    # 1. Xử lý Items Meta - Ép kiểu "vòng" để tránh lỗi Decimal -> Float64
    items_meta = lf_items.select([
        pl.col('item_id').cast(pl.String), 
        pl.col('brand').fill_null("unknown"), 
        pl.col('age_group').fill_null("unknown"), 
        pl.col('category').fill_null("unknown"), 
        # Ép kiểu Decimal -> String -> Float64 để triệt tiêu lỗi Schema Parquet
        pl.col('price').cast(pl.String).cast(pl.Float64).alias('item_price')
    ])

    # 2. Xử lý Purchases - Đảm bảo item_id là String để Join
    purchases_clean = lf_purchases.with_columns([
        pl.col('item_id').cast(pl.String),
        pl.col('customer_id').cast(pl.Int32)
    ])

    # 3. JOIN
    data_lf = purchases_clean.join(
        items_meta,
        on='item_id',
        how='left'
    )

    # 4. TÍNH TOÁN 5 FEATURES (3 CŨ + 2 MỚI)
    feature_df = (
        data_lf
        .select([
            pl.col("customer_id").alias("X_-1"),
            pl.col("item_id").alias("X_0"),

            # --- 3 FEATURE CŨ (X1, X2, X3) ---
            pl.len().over(["customer_id", "brand"]).alias("X_1"),
            pl.len().over(["customer_id", "age_group"]).alias("X_2"),
            pl.len().over(["customer_id", "category"]).alias("X_3"),

            # --- 2 FEATURE MỚI (X4, X5) ---
            # X_4: Price Affinity (Giá TB / Giá món hiện tại)
            (pl.col("item_price").mean().over("customer_id") / (pl.col("item_price") + 0.0001)).alias("X_4"),

            # X_5: Item Popularity (Tần suất xuất hiện toàn sàn)
            pl.len().over("item_id").log1p().alias("X_5"),
        ])
        # Loại bỏ trùng lặp trước khi cast cuối cùng
        .unique(subset=["X_-1", "X_0"]) 
        .with_columns([
            pl.col("X_1").cast(pl.Float64).fill_null(0),
            pl.col("X_2").cast(pl.Float64).fill_null(0),
            pl.col("X_3").cast(pl.Float64).fill_null(0),
            pl.col("X_4").cast(pl.Float64).fill_null(0),
            pl.col("X_5").cast(pl.Float64).fill_null(0),
        ])
    )

    return feature_df


def build_features_from_purchases_2(lf_purchases, lf_items, lf_users=None):
    """
    Phiên bản tối giản: Chỉ tính toán X_1, X_2, X_3 dựa trên tần suất tương tác.
    """
    
    # 1. JOIN THÔNG TIN CẦN THIẾT (Brand, Age_group, Category)
    data_lf = lf_purchases.join(
        lf_items.select(['item_id', 'brand', 'age_group', 'category']),
        on='item_id',
        how='inner'
    )

    # 2. TÍNH TOÁN USER–ITEM LEVEL FEATURES (X_1, X_2, X_3)
    feature_df = (
        data_lf
        .select([
            pl.col("customer_id").alias("X_-1"),
            pl.col("item_id").alias("X_0"),

            # X_1: User-Brand frequency
            pl.len().over(["customer_id", "brand"]).alias("X_1"),

            # X_2: User-Age_group frequency
            pl.len().over(["customer_id", "age_group"]).alias("X_2"),

            # X_3: User-Category frequency
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


def build_labels(
    baseHist: pl.LazyFrame,
    labelHist: pl.LazyFrame,
    items: pl.LazyFrame,
    negative_ratio: float = 1.0,
    seed: int = 42,
    topk_popular: int = 5000,
    max_neg_per_user: int = 50
) -> pl.LazyFrame:

    t0 = time.time()
    print("=" * 70)
    print("BUILD LABELS")
    print("=" * 70)

    # 1. POSITIVE SAMPLES
    print("Step 1 — Loading positives...")
    pos = (
        labelHist
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
        .collect()
    )
    pos = pos.with_columns([
        pl.col("customer_id").cast(pl.Int32),
        pl.col("label").cast(pl.Int8)
    ])
    print(f" → Positives: {pos.height:,}")

    # 2. POSITIVE COUNT PER USER
    pos_cnt = (
        pos
        .group_by("customer_id")
        .agg(pl.len().alias("k_pos"))
    )

    # 3. PURCHASE HISTORY MAP
    print("Step 3 — Building exclusion map...")
    hist_all = (
        pl.concat([baseHist, labelHist])
        .select(["customer_id", "item_id"])
        .unique()
        .collect()
    )

    purchase_map = (
        hist_all
        .group_by("customer_id")
        .agg(pl.col("item_id").alias("items"))
        .to_dict(as_series=False)
    )

    purchase_map = {
        u: set(it)
        for u, it in zip(purchase_map["customer_id"], purchase_map["items"])
    }

    # 4. POPULAR ITEMS (NEGATIVE POOL)
    print("Step 4 — Loading popular items...")
    popular_items = (
        baseHist
        .group_by("item_id")
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .head(topk_popular)
        .select("item_id")
        .collect()["item_id"]
        .to_list()
    )
    popular_items = np.array(popular_items)

    # 5. NEGATIVE SAMPLING
    print("Step 5 — Sampling negatives...")
    np.random.seed(seed)
    negatives = []

    for i, row in enumerate(pos_cnt.iter_rows(named=True)):
        cid = row["customer_id"]
        k_pos = row["k_pos"]

        purchased = purchase_map.get(cid, set())
        candidates = popular_items[~np.isin(popular_items, list(purchased))]

        if len(candidates) == 0: continue

        n_neg = min(int(k_pos * negative_ratio), max_neg_per_user, len(candidates))
        sampled = np.random.choice(candidates, size=n_neg, replace=False)

        for it in sampled:
            negatives.append((cid, it, 0))

        if i % 100_000 == 0 and i > 0:
            print(f"   Processed {i:,} users...")

    neg_df = pl.DataFrame(
        negatives,
        schema={
            "customer_id": pl.Int32,
            "item_id": pos.schema["item_id"],
            "label": pl.Int8
        },
        orient="row"
    )

    # 6. FINALIZING
    print("Step 6 — Finalizing dataset...")
    result = pl.concat([pos, neg_df], how="vertical")
    result = result.sample(fraction=1.0, shuffle=True, seed=seed)

    print(f"Total time: {time.time() - t0:.2f}s")
    print("=" * 70)

    return result.lazy()