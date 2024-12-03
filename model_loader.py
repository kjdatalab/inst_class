import os
import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        try:
            # Check file extension
            if self.model_path.endswith('.keras'):
                print(f"Loading Keras model from: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
            elif os.path.isdir(self.model_path):
                print(f"Loading SavedModel from directory: {self.model_path}")
                self.model = tf.saved_model.load(self.model_path)
            else:
                raise ValueError("Unsupported model format. Must be .keras file or SavedModel directory")

            return self.model

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, input_data):
        """
        Makes predictions using the loaded model.
        Handles differences in how predictions are made between the two formats.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Handle SavedModel format
            if isinstance(self.model, tf.saved_model.SavedModel):
                results = self.model.signatures['serving_default'](input_data)
                return results['classifier']  # Adjust output key as needed

            # Handle HDF5/Keras model format
            else:
                return self.model.predict(input_data)

        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

    def predict_with_confidence(self, input_data):
        """
        Makes predictions and returns both the class prediction and confidence score.
        The confidence score represents how sure the model is about its prediction.

        For binary classification (piano/no piano):
        - Returns probability for the predicted class
        - For example, if model is 0.8 confident there's a piano, returns (True, 0.8)
        - If model is 0.7 confident there's no piano, returns (False, 0.7)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Handle SavedModel format
            if isinstance(self.model, tf.saved_model.SavedModel):
                results = self.model.signatures['serving_default'](input_data)
                # Apply softmax to get probabilities
                probabilities = tf.nn.softmax(results['classifier']).numpy()

            # Handle HDF5/Keras model format
            else:
                # Keras models might output probabilities directly
                probabilities = self.model.predict(input_data)

            # Get the predicted class (index of highest probability)
            predicted_class = np.argmax(probabilities)
            # Get the confidence score for that prediction
            confidence = float(probabilities[predicted_class])

            # Convert to boolean for piano detection (assuming 1 = piano)
            contains_piano = bool(predicted_class)

            return contains_piano, confidence

        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
