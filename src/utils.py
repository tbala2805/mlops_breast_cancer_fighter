def get_logger():
    import logging
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s - %(asctime)s', level=logging.DEBUG)
    return logging


