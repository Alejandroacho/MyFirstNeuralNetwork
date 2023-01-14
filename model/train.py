from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from dataset.inputs import inputs
from dataset.outputs import outputs


def get_model() -> Sequential:
    layer: Dense = Dense(units=1, input_shape=[1])
    model: Sequential = Sequential([layer])
    model.compile(optimizer=Adam(0.1), loss='mean_squared_error')
    model.fit(inputs, outputs, epochs=1000, verbose=False)
    return model

def save_model(model: Sequential) -> None:
    model.save("model.h5")

def create_model() -> None:
    model: Sequential = get_model()
    save_model(model)

if __name__ == "__main__":
    create_model()
