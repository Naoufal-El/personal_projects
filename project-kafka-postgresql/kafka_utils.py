from confluent_kafka.admin import AdminClient, NewTopic

# Kafka broker and admin client configuration
bootstrap_servers = 'localhost:9092'
admin_client_config = {'bootstrap.servers': bootstrap_servers}

# Create a topic in Kafka
def create_topic(topic_name, num_partitions, replication_factor):
    # Create an AdminClient instance
    admin_client = AdminClient(admin_client_config)

    # Create the topic configuration
    topic_config = {'cleanup.policy': 'delete'}

    # Create the NewTopic object
    new_topic = NewTopic(topic=topic_name, num_partitions=num_partitions, replication_factor=replication_factor, config=topic_config)

    # Create the topic
    admin_client.create_topics([new_topic])

    # Close the AdminClient
    admin_client = None

# Define the topic details
topic_name = 'Doc'
num_partitions = 4
replication_factor = 2

# Create the topic
create_topic(topic_name, num_partitions, replication_factor)