import streamlit as st
import pandas as pd
from kafka import KafkaConsumer
import matplotlib.pyplot as plt
import seaborn as sns
from consumer import KafkaJSONConsumer 
from datetime import datetime
import time

# Функция для чтения данных из Kafka
def read_from_kafka(consumer):
    data = []
    try:
        while True:
            message = consumer.consume_messages()
            data.append(message)
            if len(data) >= 5:  # Считываем последние 25 сообщений для примера
                break
    except:
        pass
    return data

# Функция для выделения максимальных значений в каждом столбце, кроме первых двух
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

# Заголовок приложения
st.title('Crabs Analysis Dashboard')

# Отображение времени последнего обновления
now = datetime.now()
dt_string = now.strftime("%d %B %Y %H:%M:%S")
st.write(f"Last update: {dt_string}")

# Конфигурация параметров автообновления
if not "sleep_time" in st.session_state:
    st.session_state.sleep_time = 25

if not "auto_refresh" in st.session_state:
    st.session_state.auto_refresh = True

with st.expander("Configure Dashboard", expanded=True):
    left, right = st.columns(2)

    with left:
        auto_refresh = st.checkbox('Auto Refresh?', st.session_state.auto_refresh)

        if auto_refresh:
            number = st.number_input('Refresh rate in seconds', value=st.session_state.sleep_time)
            st.session_state.sleep_time = number

# Параметры Kafka
bootstrap_servers = 'localhost:9092' 
topic = 'result'
consumer = KafkaJSONConsumer(bootstrap_servers, topic)

# Получение и отображение данных из Kafka
st.header("Live Kafka Predictions")
data = read_from_kafka(consumer)
df = pd.DataFrame(data)
print(df)

if not df.empty:
    # Применение стиля к DataFrame
    styled_df = df.style.apply(highlight_max, subset=df.columns[2:])
    st.dataframe(styled_df)

    # Гистограмма для 'Age'
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], ax=ax)
    st.pyplot(fig)

    # Корреляционная матрица
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots()
    sns.heatmap(df.drop(columns=['id']).corr(), annot=True, ax=ax)
    st.pyplot(fig)

    # Выбор метрики для визуализации
    option = st.selectbox('Choose metric for visualization', ('Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight'))

    if option in df:
        # Гистограмма для выбранной метрики
        st.subheader(f'Distribution of {option}')
        fig, ax = plt.subplots()
        ax.hist(df[option], bins=10, color='skyblue', edgecolor='black')
        ax.set_xlabel(option)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Автообновление данных
if auto_refresh:
    time.sleep(number)
    st.experimental_rerun()
