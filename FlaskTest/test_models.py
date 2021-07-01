import tensorflow as tf

def load_model():
    loaded_model = tf.keras.models.load_model(r'C:\Users\asjad\Downloads\CV_model.pb') # change to assets

    return loaded_model

if __name__ == '__main__':
    model = load_model()
    print(model.summary())