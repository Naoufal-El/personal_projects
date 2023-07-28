from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from db_alchemy import Base, engine, SessionLocal
from pydn_schemas import Customer, Order, Product, Stock
from customer import get_customer, get_customers, create_customer, update_customer, delete_customer
from order import get_order, get_orders, create_order, update_order, delete_order
from stock import get_stock, get_stocks, create_stock, update_stock, delete_stock
from product import get_product, get_products, create_product, update_product, delete_product, get_product_stock


app = FastAPI()

# Create database tables (if they don't exist)
Base.metadata.create_all(bind=engine)


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Customer endpoints
@app.get("/customers/{cust_id}", response_model=Customer)
def read_customer(cust_id: int, db: Session = Depends(get_db)):
    customer = get_customer(db, cust_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/customers", response_model=list[Customer])
def read_customers(db: Session = Depends(get_db)):
    customers = get_customers(db)
    return customers


@app.post("/customers", response_model=Customer)
def create_new_customer(customer: Customer, db: Session = Depends(get_db)):
    return create_customer(db, customer)


@app.put("/customers/{cust_id}", response_model=Customer)
def update_existing_customer(cust_id: int, updated_data: Customer, db: Session = Depends(get_db)):
    customer = get_customer(db, cust_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return update_customer(db, customer, updated_data)


@app.delete("/customers/{cust_id}", response_model=Customer)
def delete_existing_customer(cust_id: int, db: Session = Depends(get_db)):
    customer = get_customer(db, cust_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return delete_customer(db, customer)


# Order endpoints
@app.get("/orders/{order_id}", response_model=Order)
def read_order(cust_id: int, db: Session = Depends(get_db)):
    order = get_order(db, cust_id)
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.get("/orders", response_model=list[Order])
def read_orders(db: Session = Depends(get_db)):
    orders = get_orders(db)
    return orders


@app.post("/orders_check", response_model=Order)
def create_order_if_stock(order: Order, db: Session = Depends(get_db)):
    product = get_product(db, order.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    stock = get_product_stock(db, order.product_id)
    if stock is None or stock.quantity == 0:
        raise HTTPException(status_code=400, detail="Product out of stock")
    
    return create_order(db, order)


@app.delete("/orders/{order_id}", response_model=Order)
def delete_existing_order(cust_id: int, db: Session = Depends(get_db)):
    order = get_order(db, cust_id)
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return delete_order(db, order)


# Stock endpoints
@app.get("/stocks/{stock_id}", response_model=Stock)
def read_stock(cust_id: int, db: Session = Depends(get_db)):
    stock = get_stock(db, cust_id)
    if stock is None:
        raise HTTPException(status_code=404, detail="Stock not found")
    return stock


@app.get("/stocks", response_model=list[Stock])
def read_stocks(db: Session = Depends(get_db)):
    stocks = get_stocks(db)
    return stocks


@app.post("/stocks_check", response_model=Stock)
def create_new_stock(stock: Stock, db: Session = Depends(get_db)):
    existing_stock = get_stock(db, stock.stock_id)
    if existing_stock is not None:
        raise HTTPException(status_code=400, detail="Stock ID already exists")
    
    return create_stock(db, stock)


@app.put("/stocks/{stock_id}", response_model=Stock)
def update_existing_stock(cust_id: int, updated_data: Stock, db: Session = Depends(get_db)):
    stock = get_stock(db, cust_id)
    if stock is None:
        raise HTTPException(status_code=404, detail="Stock not found")
    return update_stock(db, stock, updated_data)


@app.delete("/stocks/{stock_id}", response_model=Stock)
def delete_existing_stock(cust_id: int, db: Session = Depends(get_db)):
    stock = get_stock(db, cust_id)
    if stock is None:
        raise HTTPException(status_code=404, detail="Stock not found")
    return delete_stock(db, stock)


# Product endpoints
@app.get("/products/{product_id}", response_model=Product)
def read_product(product_id: int, db: Session = Depends(get_db)):
    product = get_product(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/products", response_model=list[Product])
def read_products(db: Session = Depends(get_db)):
    products = get_products(db)
    return products


@app.post("/products", response_model=Product)
def create_new_product(product: Product, db: Session = Depends(get_db)):
    return create_product(db, product)


@app.put("/products/{product_id}", response_model=Product)
def update_existing_product(product_id: int, updated_data: Product, db: Session = Depends(get_db)):
    product = get_product(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return update_product(db, product, updated_data)


@app.delete("/products/{product_id}", response_model=Product)
def delete_existing_product(product_id: int, db: Session = Depends(get_db)):
    product = get_product(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return delete_product(db, product)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)