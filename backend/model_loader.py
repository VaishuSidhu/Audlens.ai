import os


def _make_compat_batch_norm():
    """
    Returns a BatchNormalization subclass that strips legacy kwargs
    (renorm, renorm_clipping, renorm_momentum) that were removed in Keras 3.x.
    This allows loading .h5 models saved with TF 2.x / Keras 2.x.
    """
    try:
        from tensorflow import keras

        class CompatBatchNormalization(keras.layers.BatchNormalization):
            # Keys that Keras 3.x no longer accepts but may appear in old .h5 configs
            _LEGACY_KEYS = {"renorm", "renorm_clipping", "renorm_momentum"}

            def __init__(self, **kwargs):
                for key in list(self._LEGACY_KEYS):
                    kwargs.pop(key, None)
                super().__init__(**kwargs)

            @classmethod
            def from_config(cls, config):
                for key in list(cls._LEGACY_KEYS):
                    config.pop(key, None)
                return cls(**config)

        return CompatBatchNormalization
    except Exception:
        return None


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

                model_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), '..', 'model', 'models', 'deepfake_hybrid_v2.h5'
                ))

                # Import custom objects for loading
                from model.models.hybrid_model import SelfAttention

                # CompatBatchNormalization strips renorm kwargs unsupported by Keras 3.x
                CompatBN = _make_compat_batch_norm()

                custom_objects = {
                    'SelfAttention': SelfAttention,
                    'focal_loss_fixed': lambda y_true, y_pred: y_pred,  # Minimal stub for loading
                }
                if CompatBN is not None:
                    custom_objects['BatchNormalization'] = CompatBN

                cls._model = keras.models.load_model(
                    model_path, custom_objects=custom_objects, compile=False
                )
                print("Model loaded successfully.")
            except Exception as e:
                print(f"CRITICAL: Error loading model: {e}")
                raise RuntimeError(f"Failed to load the deepfake detection model: {str(e)}")

        return cls._model

