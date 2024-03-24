from kafka import KafkaProducer

class KafkaCSVProducer:
    def __init__(self, bootstrap_servers, topic):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.topic = topic

    def delivery_report(self, err, msg):
        if err is not None:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic} [{msg.partition}] offset {msg.offset}')

    def send_csv_data(self, data):
        future = self.producer.send(self.topic, str(data).encode('utf-8'))
        try:
            record_metadata = future.get(timeout=10)
            self.delivery_report(None, record_metadata)
        except Exception as e:
            self.delivery_report(e, None)

        self.producer.flush()