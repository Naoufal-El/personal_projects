from avro_utils import initialize_avro_producer, serialize_avro_message
from postgres_utils import connect_to_postgres, listen_to_table_changes, retrieve_changed_row
from confluent_kafka import avro
from offset_utils import load_last_offset, save_last_offset
import time

# Kafka configuration
bootstrap_servers = 'localhost:9092'
topic = 'Documents'

# Avro schema and serialization configuration
avro_schema_str = '''
    {
        "type": "record",
        "name": "Document",
        "fields": [
            {"name": "doc_id", "type": "int"},
            {"name": "doc_type", "type": "string"},
            {"name": "doc_author", "type": "string"}
        ]
    }
'''
avro_schema = avro.loads(avro_schema_str)

# PostgreSQL configuration
db_host = 'localhost'
db_port = 5432
db_name = 'document_exkfk'
db_user = 'postgres'
db_password = 'admin12'



def publish_message(producer, key, value):
    try:
        producer.produce(topic=topic, key=key, value=value)
        producer.flush()
        print(f"Published message: {value}")
    except Exception as e:
        print(f"Failed to publish message: {value}\nError: {e}")


def main():

    # Initialize Avro producer
    avro_producer = initialize_avro_producer(bootstrap_servers, avro_schema)

    # Connect to PostgreSQL
    connection = connect_to_postgres(db_host, db_port, db_name, db_user, db_password)
    cursor = connection.cursor()

    # Load last offset
    last_offset = load_last_offset()

    # Start consuming and publishing changes
    while True:
        # Listen to table changes
        listen_to_table_changes(connection, cursor, 'doc_logs')

        # Retrieve changed rows
        cursor.execute("SELECT * FROM doc_logs WHERE log_id > %s;", (last_offset,))
        rows = cursor.fetchall()

        for row in rows:
            doc_id = row[0]
            doc_data = retrieve_changed_row(cursor, 'doc_logs', doc_id)
            avro_message = serialize_avro_message(avro_schema, doc_data)
            publish_message(avro_producer, str(doc_id), avro_message)

            # Update last offset
            last_offset = doc_id

        # Save last offset
        save_last_offset(last_offset)

        # Sleep for 60 seconds before checking for new changes
        time.sleep(60)


if __name__ == '__main__':
    main()



