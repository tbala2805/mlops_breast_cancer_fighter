import pandas as pd
import os
import plotly.express as px


def perform_eda(data: pd.DataFrame, eda_path: str, target_column:str )-> None:

    # writing info
    with open(os.path.join(eda_path, "data_information.txt"), 'w') as f:
        data.info(buf=f)

    data_diagnosis = data[target_column].value_counts().reset_index()
    fig = px.pie(data_diagnosis, values='count', names='diagnosis', title='the number of Malignant and Benign')
    fig.write_image(os.path.join(eda_path, "pie_chart.png"))




