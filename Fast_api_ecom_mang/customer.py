from db_alchemy import Customer


# CRUD operations in data base

def get_customer(db, cust_id):
    return db.query(Customer).filter(Customer.cust_id == cust_id).first()


def get_customers(db):
    return db.query(Customer).all()


def create_customer(db, customer):
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


def update_customer(db, customer, updated_data):
    for key, value in updated_data.items():
        setattr(customer, key, value)
    db.commit()
    db.refresh(customer)
    return customer


def delete_customer(db, customer):
    db.delete(customer)
    db.commit()
    return customer