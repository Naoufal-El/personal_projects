from db_alchemy import Stock


# CRUD operations

def get_stock(db, cust_id):
    return db.query(Stock).filter(Stock.cust_id == cust_id).first()


def get_stocks(db):
    return db.query(Stock).all()


def create_stock(db, stock):
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


def update_stock(db, stock, updated_data):
    for key, value in updated_data.items():
        setattr(stock, key, value)
    db.commit()
    db.refresh(stock)
    return stock


def delete_stock(db, stock):
    db.delete(stock)
    db.commit()
    return stock