from confluent_kafka.avro import AvroProducer, AvroConsumer


def initialize_avro_producer(bootstrap_servers, avro_schema):
    producer_config = {
        'bootstrap.servers': bootstrap_servers,
        'schema.registry.url': 'http://localhost:8081'
    }
    return AvroProducer(producer_config, default_value_schema=avro_schema)


def initialize_avro_consumer(bootstrap_servers, topic, avro_schema, offset):
    consumer_config = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': 'doc_logs_group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': 'false',
        'schema.registry.url': 'http://localhost:8081'
    }
    consumer = AvroConsumer(consumer_config)
    consumer.subscribe([topic])
    consumer.poll(0)
    consumer.seek(topic, 0, offset)  # Seek to the last committed offset
    return consumer


def serialize_avro_message(avro_schema, row):
    avro_message = {
        'doc_id': row[0],
        'doc_type': row[1],
        'doc_author': row[2]
    }
    return avro_message
