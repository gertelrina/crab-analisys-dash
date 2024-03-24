
FROM python:3.8


ADD get_data.py /lab/
ADD sent_data.py /lab/
ADD producer.py /lab/
ADD consumer.py /lab/
ADD requirements.txt /lab/
ADD streamlit_app.py /lab/


ADD runner.sh /lab/

COPY ./data /lab/data
COPY ./weights /lab/weights


WORKDIR /lab/
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /log/