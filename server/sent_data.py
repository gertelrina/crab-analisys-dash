from producer import KafkaCSVProducer
import time
import pandas as pd

bootstrap_servers = 'localhost:9092'
topic = 'raw_data'
producer = KafkaCSVProducer(bootstrap_servers, topic)

test = pd.read_csv('/lab/data/test.csv')
print('The dimension of the test synthetic dataset is:', test.shape)

for index, row in test.iterrows():
    time.sleep(2)
    data = row.to_dict()
    producer.send_csv_data(data)

