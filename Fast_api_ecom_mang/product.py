from db_alchemy import Product

def get_product(db, product_id):
    return db.query(Product).filter(Product.product_id == product_id).first()


def get_products(db):
    return db.query(Product).all()


def create_product(db, Product):
    db.add(Product)
    db.commit()
    db.refresh(Product)
    return Product


def update_product(db, Product, updated_data):
    for key, value in updated_data.items():
        setattr(Product, key, value)
    db.commit()
    db.refresh(Product)
    return Product


def delete_product(db, Product):
    db.delete(Product)
    db.commit()
    return Product


def get_product_stock(db, product_id):
    product = get_product(db, product_id)
    if product and product.stock:
        return product.stock
    return []