import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib 
from consumer import KafkaJSONConsumer
from producer import KafkaCSVProducer
import time

# Preprocessing Function
def preprocess_data(data_dict):
    df = pd.DataFrame([data_dict])
    df['generated'] = 1  # Add 'generated' column

    # Encoding categorical features
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    return df

if __name__ == "__main__":
    bootstrap_servers = 'localhost:9092'

    topic_res = 'result'
    producer = KafkaCSVProducer(bootstrap_servers, topic_res)

    topic = 'raw_data'
    consumer = KafkaJSONConsumer(bootstrap_servers, topic)

    gb_model = joblib.load('/lab/weights/hist_md.joblib')
    while True:
        info = consumer.consume_messages()
        print(info)
        df = preprocess_data(info)
        test_baseline = df.drop(columns=['id'])

    # # Predictions
        gb_predictions = gb_model.predict(test_baseline)

    # # Preparing the answer
        info['Age'] = gb_predictions[0]
        producer.send_csv_data(info)