from pydantic import BaseModel
from datetime import date


class Customer(BaseModel):
    cust_id: int
    cust_name: str
    cust_address: str
    cust_tel: int


class Order(BaseModel):
    order_id: int
    cust_id: int
    cust_name: str
    product_id: int
    amount: float
    order_date: date


class Product(BaseModel):
    product_id: int
    product_price: float
    product_type: str


class Stock(BaseModel):
    stock_id: int
    product_id: int
    quantity: float
    shop_no: int
