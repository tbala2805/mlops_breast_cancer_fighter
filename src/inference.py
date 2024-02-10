import tensorflow as tf


def _load_model(model_path):
    """function to load the model"""

    return tf.keras.models.load_model(model_path)


def inference_load_model_dev(model, data):
    "function to perform inference on the model"

    return model.predict(data)


def preprocessing_inference(min_max_scaler, data, columns_to_drop):
    # Preprocessing a bit
    data.drop(columns=columns_to_drop, inplace=True, axis=1)
    return min_max_scaler.transform(data)



