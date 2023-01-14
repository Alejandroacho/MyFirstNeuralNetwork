from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model


def execute_model() -> None:
    while True:
        user_input: str or int = input("Enter a number ('q' for quit): ")
        if user_input == "q":
            break
        try:
            number: float = float(user_input)
        except ValueError:
            continue
        print(get_result(number))

def get_result(number: float) -> list:
    model: Sequential = load_model("model.h5")
    result: list = model.predict([number])
    return result

if __name__ == "__main__":
    execute_model()
