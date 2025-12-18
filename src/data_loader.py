import polars as pl
import glob
import os
import datetime as dt # Import datetime for date objects
import polars as pl  


def load_parquet_files(data_path, prefix):
    search_pattern = os.path.join(data_path, f"{prefix}*.parquet")
    files = glob.glob(search_pattern)
    if not files:
        print(f"⚠️ Cảnh báo: Không tìm thấy file nào với tiền tố '{prefix}' tại {data_path}")
        return None
    return pl.concat([pl.scan_parquet(f) for f in files])

def get_all_data(data_path="data"):
    print(f"📂 Đang quét dữ liệu từ thư mục: {os.path.abspath(data_path)}")
    
    # Load các bảng dựa trên tiền tố file của bạn
    lf_items = load_parquet_files(data_path, "sales_pers.item_chunk_")
    lf_users = load_parquet_files(data_path, "sales_pers.user_chunk_")
    lf_purchases = load_parquet_files(data_path, "sales_pers.purchase_history_daily_chunk_")
    
    # Kiểm tra nhanh số lượng (Lưu ý: collect() tốn tài nguyên, chỉ nên dùng khi debug)
    if lf_items is not None:
        print(f"✅ Đã load Items")
    if lf_users is not None:
        print(f"✅ Đã load Users")
    if lf_purchases is not None:
        print(f"✅ Đã load Purchases")
        
    return lf_items, lf_users, lf_purchases

def split_date_lazy(
    lf: pl.LazyFrame,
    date_column_name: str,
    hist_end: dt.datetime = dt.datetime(2024, 10, 31, 23, 59, 59),
    recent_end: dt.datetime = dt.datetime(2024, 11, 30, 23, 59, 59)
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    schema = lf.collect_schema()
    data_dtype = schema.get(date_column_name, None)

    if data_dtype not in (pl.Date, pl.Datetime):
        lf = lf.with_columns(
            pl.col(date_column_name)
            .str.to_datetime(strict=False)
            .alias(date_column_name)
        )
    lf_hist = lf.filter(
        pl.col(date_column_name) <= hist_end
    )
    lf_recent = lf.filter(
        (pl.col(date_column_name) > hist_end) &
        (pl.col(date_column_name) <= recent_end)
    )
    lf_val = lf.filter(
        pl.col(date_column_name) > recent_end
    )
    return lf_hist, lf_recent, lf_val