from kafka_utils import create_topic
from avro_utils import initialize_avro_consumer
from confluent_kafka import avro


# Kafka configuration
bootstrap_servers = 'localhost:9092'
topic = 'documents'

# Avro schema and deserialization configuration
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

# Offset file
offset_file = 'last_offset.txt'


def load_last_offset():
    try:
        with open(offset_file, 'r') as file:
            return int(file.read())
    except FileNotFoundError:
        return 0


def save_last_offset(offset):
    with open(offset_file, 'w') as file:
        file.write(str(offset))


def deserialize_avro_message(avro_schema, message):
    doc_id = message['doc_id']
    doc_type = message['doc_type']
    doc_author = message['doc_author']
    return doc_id, doc_type, doc_author


def consume_messages(consumer):
    while True:
        message = consumer.poll(1.0)

        if message is None:
            continue

        if message.error():
            print(f"Consumer error: {message.error()}")
            continue

        doc_id, doc_type, doc_author = deserialize_avro_message(avro_schema, message.value())
        print(f"Received message: doc_id={doc_id}, doc_type={doc_type}, doc_author={doc_author}")

        # Commit the offset manually
        consumer.commit(message)

        # Save the last offset
        save_last_offset(message.offset() + 1)


def main():
    # Create Kafka topic
    create_topic(bootstrap_servers, topic)

    # Initialize Avro consumer
    last_offset = load_last_offset()
    consumer = initialize_avro_consumer(bootstrap_servers, topic, avro_schema, last_offset)

    # Consume messages
    consume_messages(consumer)


if __name__ == '__main__':
    main()
