from kafka import KafkaConsumer
import json

class KafkaJSONConsumer:
    def __init__(self, bootstrap_servers, topic, group_id=None):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8').replace("'", "\"")))
        

    def consume_messages(self):
        for message in self.consumer:
            print(f"Received message: {message.value}")
            return message.value
