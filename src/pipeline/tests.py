# test_predict_pipeline.py

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def test_prediction():
    # Define class labels
    class_labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
        3: "Spam",
        4: "Urgent",
        5: "Other"
    }

    # Initialize the prediction pipeline
    predict_pipeline = PredictPipeline(model_dir="artifact/model_tf", class_labels=class_labels)

    # Test inputs
    test_inputs = [
        "I love this product!",
        "This is terrible.",
        "Please respond urgently.",
        "Buy now! Limited offer.",
        " ",
        None
    ]

    for text in test_inputs:
        try:
            if text is None:
                print("Input text is None. Skipping prediction.")
                continue

            # Create CustomData instance
            data = CustomData(text=text)

            # Get data as list
            data_as_list = data.get_data_as_list()

            # Make prediction
            predicted_label = predict_pipeline.predict(data_as_list)[0]

            print(f"Input Text: '{text}'")
            print(f"Predicted Class: {predicted_label}\n")

        except Exception as e:
            print(f"Error predicting for input '{text}': {e}\n")

if __name__ == "__main__":
    test_prediction()
