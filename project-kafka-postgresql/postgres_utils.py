import psycopg2


def connect_to_postgres(db_host, db_port, db_name, db_user, db_password):
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )


def listen_to_table_changes(connection, cursor, table_name):
    cursor.execute(f"LISTEN {table_name}_changes;")
    connection.commit()
    print(f"Monitoring {table_name} table for changes...")


def retrieve_changed_row(cursor, table_name, doc_id):
    cursor.execute(f"SELECT * FROM {table_name} WHERE doc_id = %s;", (doc_id,))
    return cursor.fetchone()
