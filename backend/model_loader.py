import os

class ModelLoader:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        """
        Loads the TensorFlow/Keras model once and reuses it for all subsequent requests.
        """
        if cls._model is None:
            print("Loading deepfake detection model...")
            try:
                from tensorflow import keras
                import tensorflow as tf
                
                # Optional: limit memory usage
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        print(e)
                        
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'models', 'deepfake_hybrid_v2.h5'))
                
                # Import custom objects for loading
                from model.models.hybrid_model import SelfAttention
                
                # We define a dummy focal loss for loading (or use the one from hybrid_model)
                custom_objects = {
                    'SelfAttention': SelfAttention,
                    'focal_loss_fixed': lambda y_true, y_pred: y_pred # Minimal stub for loading
                }
                
                cls._model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"CRITICAL: Error loading model: {e}")
                raise RuntimeError(f"Failed to load the deepfake detection model: {str(e)}")
                
        return cls._model
