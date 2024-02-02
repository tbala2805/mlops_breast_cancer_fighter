from src.training import initializing_model, fitting_model
from src.preprocessing import preprocess
from src.eda import perform_eda
import pandas as pd

# need to add config
path = "./data/raw/data.csv"
data = pd.read_csv(path)

X, y = preprocess(data, columns_to_drop=['id', "Unnamed: 32"],
                  target_col='diagnosis')
print(X, y)
perform_eda(data, eda_path='./report/eda_report', target_column='diagnosis')


model = initializing_model()
#
fitting_model(model, X, y)
