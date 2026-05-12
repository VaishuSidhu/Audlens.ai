import os


class DummyDTypePolicy(str):
    """
    Acts as a string-like dtype object to resolve Keras 3 serialized DTypePolicy
    dictionaries when deserialized in legacy Keras environments.
    """
    def __new__(cls, *args, **kwargs):
        name = "float32"
        if args and isinstance(args[0], str):
            name = args[0]
        elif "name" in kwargs:
            name = kwargs["name"]
        elif args and isinstance(args[0], dict):
            name = args[0].get("name", "float32")
        obj = super().__new__(cls, name)
        obj.name = name
        obj.compute_dtype = name
        obj.variable_dtype = name
        obj.quantization_mode = None
        return obj

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(config.get("name", "float32"))
        return cls(str(config))

    def get_config(self):
        return {"name": getattr(self, "name", str(self))}


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


def _make_compat_input_layer():
    """
    Returns an InputLayer subclass that strips legacy kwargs
    (batch_shape, optional) for compatibility when loading old configs.
    """
    try:
        from tensorflow import keras

        class CompatInputLayer(keras.layers.InputLayer):
            def __init__(self, **kwargs):
                if "batch_shape" in kwargs:
                    shape = kwargs.pop("batch_shape")
                    if shape and len(shape) > 1 and kwargs.get("input_shape") is None:
                        kwargs["input_shape"] = shape[1:]
                kwargs.pop("optional", None)
                super().__init__(**kwargs)

            @classmethod
            def from_config(cls, config):
                cfg = dict(config)
                if "batch_shape" in cfg:
                    shape = cfg.pop("batch_shape")
                    if shape and len(shape) > 1 and cfg.get("input_shape") is None:
                        cfg["input_shape"] = shape[1:]
                cfg.pop("optional", None)
                return cls(**cfg)

        return CompatInputLayer
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

                # Compat layers strip unsupported legacy kwargs
                CompatBN = _make_compat_batch_norm()
                CompatInput = _make_compat_input_layer()

                # Monkey-patch InputLayer.from_config globally to handle internal deserialization paths
                if hasattr(keras.layers.InputLayer, 'from_config'):
                    orig_input_from_config = keras.layers.InputLayer.from_config
                    if not getattr(orig_input_from_config, '_is_patched', False):
                        def patched_input_from_config(config):
                            cfg = dict(config)
                            if "batch_shape" in cfg:
                                shape = cfg.pop("batch_shape")
                                if shape and len(shape) > 1 and cfg.get("input_shape") is None:
                                    cfg["input_shape"] = shape[1:]
                            cfg.pop("optional", None)
                            return orig_input_from_config(cfg)
                        patched_input_from_config._is_patched = True
                        keras.layers.InputLayer.from_config = patched_input_from_config

                custom_objects = {
                    'SelfAttention': SelfAttention,
                    'focal_loss_fixed': lambda y_true, y_pred: y_pred,  # Minimal stub for loading
                    'DTypePolicy': DummyDTypePolicy,
                }
                if CompatBN is not None:
                    custom_objects['BatchNormalization'] = CompatBN
                if CompatInput is not None:
                    custom_objects['InputLayer'] = CompatInput

                cls._model = keras.models.load_model(
                    model_path, custom_objects=custom_objects, compile=False
                )
                print("Model loaded successfully.")
            except Exception as e:
                print(f"CRITICAL: Error loading model: {e}")
                raise RuntimeError(f"Failed to load the deepfake detection model: {str(e)}")

        return cls._model

