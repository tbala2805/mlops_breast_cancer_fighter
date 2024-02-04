from src.training import initializing_model, fitting_model
from src.preprocessing import preprocess
from src.eda import perform_eda
import pandas as pd
from src.utils import get_logger

log = get_logger()
log.getLogger('matplotlib.font_manager').disabled = True # disabling the matplotlib debugging for my project


# need to add config
path = "./data/raw/data.csv"
# reading file
log.info("reading a file")
data = pd.read_csv(path)
log.info(data.head(2))

log.info("preprocessing the data...")
X, y = preprocess(data, columns_to_drop=['id', "Unnamed: 32"],
                  target_col='diagnosis')
print(X, y)
log.info("Performing EDA on the data")
perform_eda(data, eda_path='./report/eda_report', target_column='diagnosis')
log.info("EDA is completed")
#
#
log.info("Initializing the tf model")
model = initializing_model()
log.info("Model is initialized")
#
log.info("Fitting a model")
fitting_model(model, X, y)
log.info("Training is completed")
