from pathlib import Path
import pandas as pd
from olist.config import RAW_DATA_DIR, INTERIM_DATA_DIR

RAW = RAW_DATA_DIR
INTERIM = INTERIM_DATA_DIR 
ORDER_DATES = [ "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", ] 

# Function: load_raw_tables()
def load_raw_tables() -> dict[str, pd.DataFrame]: 
    """Load all 9 raw Olist CSVs. Returns dict of DataFrames.""" 
    return { 
        "orders": pd.read_csv(RAW/"olist_orders_dataset.csv", parse_dates=ORDER_DATES), 
        "customers": pd.read_csv(RAW/"olist_customers_dataset.csv"), 
        "items": pd.read_csv(RAW/"olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]), 
        "payments": pd.read_csv(RAW/"olist_order_payments_dataset.csv"), 
        "reviews": pd.read_csv(RAW/"olist_order_reviews_dataset.csv", parse_dates=["review_creation_date", "review_answer_timestamp"]), 
        "products": pd.read_csv(RAW/"olist_products_dataset.csv"), 
        "sellers": pd.read_csv(RAW/"olist_sellers_dataset.csv"), 
        "geo": pd.read_csv(RAW/"olist_geolocation_dataset.csv"), 
        "cat_trans": pd.read_csv(RAW/"product_category_name_translation.csv"), 
    }

# Function: clean_orders()
def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 'delivered' orders (has all timestamp cols populated)
    Drop rows where estimated_delivery_date is null
    Assert no negative purchase timestamps
    """
    df = orders.copy()
    
    # Keep only delivered orders for analysis
    df = df[df["order_status"] == "delivered"].copy()
    
    # Drop rows missing critical timestamps
    critical_cols = [
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    print(f"clean_orders: dropped {dropped} rows with null delivery dates")
    
    # Sanity check — purchase before approval
    bad_approvals = (df["order_approved_at"] < df["order_purchase_timestamp"]).sum()
    if bad_approvals > 0:
        print(f"WARNING: {bad_approvals} orders approved before purchase")
    
    return df.reset_index(drop=True)

#Function: clean_products()

def clean_products(
    products: pd.DataFrame,
    cat_trans: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill null categories, merge English translation.
    """
    df = products.copy()
    
    # Fill null category names with 'unknown'
    df["product_category_name"] = (
        df["product_category_name"].fillna("unknown")
    )
    
    # Merge English translation
    df = df.merge(cat_trans, on="product_category_name", how="left")
    df["product_category_name_english"] = (
        df["product_category_name_english"].fillna("unknown")
    )
    
    return df

# Function: clean_geo()
def clean_geo(geo: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate geolocation by averaging lat/lng per zip prefix.
    """
    return (
        geo
        .groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            state=("geolocation_state", "first"),
        )
        .reset_index()
        .rename(columns={
            "geolocation_zip_code_prefix": "zip_prefix"
        })
    )

# Function: aggregate_payments()    
def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Payments has multiple rows per order (installments).
    Collapse to one row per order: total value + dominant payment type.
    """
    agg = (
        payments
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),  # dominant type
            n_payment_types=("payment_type", "nunique"),
        )
        .reset_index()
    )
    return agg

# Function: aggregate_items()
def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Items has one row per item (orders can have multiple items).
    Collapse to one row per order.
    """
    agg = (
        items
        .groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            seller_id=("seller_id", "first"),
            product_id=("product_id", "first"),  # ← add this line
        )
        .reset_index()
    )
    return agg

    

def build_master(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    items_agg: pd.DataFrame,
    payments_agg: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    geo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all cleaned tables into one analysis-ready DataFrame.
    One row = one delivered order.
    """
    df = orders.copy()

    df = df.merge(
        customers[[
            "customer_id", "customer_state",
            "customer_city", "customer_zip_code_prefix"
        ]],
        on="customer_id",
        how="left"
    )
    df = df.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")

    reviews_slim = reviews[["order_id", "review_score", "review_comment_message"]].copy()
    reviews_slim["is_bad_review"] = reviews_slim["review_score"] <= 2
    df = df.merge(reviews_slim, on="order_id", how="left")

    df = df.merge(
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left"
    )
    df = df.merge(
        products[["product_id", "product_category_name_english", "product_weight_g"]],
        on="product_id",
        how="left"
    )
    df = df.merge(
        geo.rename(columns={
            "zip_prefix": "customer_zip_code_prefix",
            "lat": "customer_lat",
            "lng": "customer_lng"
        }),
        on="customer_zip_code_prefix",
        how="left"
    )

    return df.reset_index(drop=True)

def save_master(df: pd.DataFrame) -> None:
    """Save master DataFrame to parquet."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DATA_DIR / "olist_master.parquet"
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Saved: {path}  ({size_mb:.1f} MB)")    