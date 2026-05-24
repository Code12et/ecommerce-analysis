import pytest
import pandas as pd
from olist.data import clean_orders, aggregate_payments, aggregate_items


def make_orders():
    return pd.DataFrame({
        "order_id": ["a1", "a2", "a3", "a4"],
        "customer_id": ["c1", "c2", "c3", "c4"],
        "order_status": ["delivered", "delivered", "canceled", "delivered"],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"]),
        "order_approved_at": pd.to_datetime([
            "2018-01-02", "2018-02-02", None, "2018-04-02"]),
        "order_delivered_carrier_date": pd.to_datetime([
            "2018-01-05", "2018-02-05", None, "2018-04-05"]),
        "order_delivered_customer_date": pd.to_datetime([
            "2018-01-10", None, None, "2018-04-10"]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2018-01-12", "2018-02-12", None, "2018-04-12"]),
    })


def make_payments():
    return pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "payment_type": ["credit_card", "voucher", "boleto"],
        "payment_value": [100.0, 20.0, 50.0],
        "payment_installments": [3, 1, 1],
    })


def make_items():
    return pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "order_item_id": [1, 2, 1],
        "seller_id": ["s1", "s1", "s2"],
        "price": [50.0, 50.0, 80.0],
        "freight_value": [10.0, 10.0, 15.0],
    })


# clean_orders tests
def test_clean_orders_keeps_only_delivered():
    df = clean_orders(make_orders())
    assert (df["order_status"] == "delivered").all()

def test_clean_orders_drops_null_delivery_dates():
    df = clean_orders(make_orders())
    assert df["order_delivered_customer_date"].isnull().sum() == 0
    assert df["order_estimated_delivery_date"].isnull().sum() == 0

def test_clean_orders_no_duplicate_order_ids():
    df = clean_orders(make_orders())
    assert df["order_id"].duplicated().sum() == 0

def test_clean_orders_correct_row_count():
    df = clean_orders(make_orders())
    assert len(df) == 2


# aggregate_payments tests
def test_aggregate_payments_one_row_per_order():
    df = aggregate_payments(make_payments())
    assert df["order_id"].duplicated().sum() == 0
    assert len(df) == 2

def test_aggregate_payments_sums_value():
    df = aggregate_payments(make_payments())
    o1 = df[df["order_id"] == "o1"].iloc[0]
    assert o1["payment_value"] == pytest.approx(120.0)

def test_aggregate_payments_correct_n_types():
    df = aggregate_payments(make_payments())
    o1 = df[df["order_id"] == "o1"].iloc[0]
    assert o1["n_payment_types"] == 2


# aggregate_items tests
def test_aggregate_items_one_row_per_order():
    df = aggregate_items(make_items())
    assert df["order_id"].duplicated().sum() == 0
    assert len(df) == 2

def test_aggregate_items_counts_items():
    df = aggregate_items(make_items())
    o1 = df[df["order_id"] == "o1"].iloc[0]
    assert o1["n_items"] == 2

def test_aggregate_items_sums_price():
    df = aggregate_items(make_items())
    o1 = df[df["order_id"] == "o1"].iloc[0]
    assert o1["total_price"] == pytest.approx(100.0)