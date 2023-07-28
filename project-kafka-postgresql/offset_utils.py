offset_file = 'C:/Users/naouf/PycharmProjects/Kafka-dock/project-kafka/last_offset.txt'

def load_last_offset():
    try:
        with open(offset_file, 'r') as file:
            return int(file.read())
    except FileNotFoundError:
        return 0


def save_last_offset(offset):
    with open(offset_file, 'w') as file:
        file.write(str(offset))