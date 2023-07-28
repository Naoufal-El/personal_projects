from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table, Date, Float
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

# Create the SQLAlchemy engine
db_url = "postgresql://postgres:admin12@localhost/api-fast"
engine = create_engine(db_url)
Base = declarative_base()

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# association table Many to Many
order_customer_table = Table('order_customer', Base.metadata,
    Column('order_id', Integer, ForeignKey('order.order_id')),
    Column('cust_id', Integer, ForeignKey('customer.cust_id'))
)


class Customer(Base):
    __tablename__ = "customer"
    cust_id = Column(Integer, primary_key=True, nullable=False)
    cust_name = Column(String, nullable=False)
    cust_address = Column(String)
    cust_tel = Column(Integer)


class Order(Base):
    __tablename__ = "order"
    order_id = Column(Integer, primary_key=True, nullable=False)
    cust_id = Column(Integer, nullable=False)
    cust_name = Column(String, nullable= False)
    product_id = Column(Integer, nullable=False)
    quantity = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    order_date = Column(Date, nullable=False)
    
    products = relationship('Product', back_populates='order')

class Product(Base):
    __tablename__ = "product"
    product_id = Column(Integer, primary_key=True, nullable=False)
    product_price = Column(Float, nullable= False)
    product_type = Column(String)
    order_id = Column(Integer, ForeignKey('order.order_id'))
    stock_id = Column(Integer, ForeignKey('stock.stock_id'))
    order = relationship("Order", back_populates="products")
    stock = relationship("Stock", back_populates="products")

class Stock(Base):
    __tablename__ = "stock"
    stock_id = Column(Integer, primary_key=True, nullable= False)
    product_id = Column(Integer,nullable= False)
    quantity = Column(Integer, nullable= False)
    shop_no = Column(Integer)

    products = relationship("Product", back_populates="stock") 

# Create the database tables (if they don't exist)
Base.metadata.create_all(engine)