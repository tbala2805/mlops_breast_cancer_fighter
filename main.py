from src.training import initializing_model, fitting_model
from src.preprocessing import preprocess
from src.eda import perform_eda
import pandas as pd
from src.utils import get_logger
import configparser

log = get_logger()
log.getLogger('matplotlib.font_manager').disabled = True # disabling the matplotlib debugging for my project


log.info('Reading configs')

configprsr = configparser.ConfigParser()
config = configprsr.read('config.ini')
path = configprsr['DEFAULT']['path']
eda_report_path = configprsr['DEFAULT']['eda_report_path']
model_report_path = configprsr['DEFAULT']['model_report_path']

cols_to_drop = configprsr['PREPROCESSING']['cols_to_drop'].split(',')
target_col = configprsr['PREPROCESSING']['target_col']

log.info(f"\nConfigs are as path: {path}\neda report path:{eda_report_path}"
         f"\nmodel_report_path: {model_report_path}\ncols_to_drp: {cols_to_drop}"
         f"\ntarget_col: {target_col}")

# reading file
log.info("reading a file")
data = pd.read_csv(path)
log.info(data.head(2))

log.info("preprocessing the data...")
X, y = preprocess(data, columns_to_drop=cols_to_drop,
                  target_col=target_col)
print(X, y)
log.info("Performing EDA on the data")
perform_eda(data, eda_path=eda_report_path, target_column=target_col)
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
