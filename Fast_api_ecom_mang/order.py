from db_alchemy import Order


def get_order(db, cust_id):
    return db.query(Order).filter(Order.cust_id == cust_id).first()


def get_orders(db):
    return db.query(Order).all()


def create_order(db, Order):
    db.add(Order)
    db.commit()
    db.refresh(Order)
    return Order


def update_order(db, Order, updated_data):
    for key, value in updated_data.items():
        setattr(Order, key, value)
    db.commit()
    db.refresh(Order)
    return Order


def delete_order(db, Order):
    db.delete(Order)
    db.commit()
    return Order