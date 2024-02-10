import tensorflow as tf


def _load_model(model_path):
    """function to load the model"""

    return tf.keras.models.load_model(model_path)


def inference_load_model_dev(model, data):
    "function to perform inference on the model"

    return model.predict(data)



