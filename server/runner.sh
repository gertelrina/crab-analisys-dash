#!/bin/bash

echo "Wait for kafka"

sleep 20

echo "Run sent data"

python3 ./sent_data.py &
sleep 2

echo "Run get data"
python3 ./get_data.py &

echo "Run get streamlit"

streamlit run streamlit_app.py&


for (( ; ; ))
do
    echo "Press CTRL+C to stop the loop."
    sleep 30
done

