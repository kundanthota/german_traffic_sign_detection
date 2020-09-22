import keras
if __name__ == "__main__":
    trained_model = keras.models.load_model(f"model/trained_model")
    score= trained_model.evaluate(x_test_gray_norm, y_test)
    print(score)