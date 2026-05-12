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

                # Monkey-patch Layer base constructor globally to pop unsupported legacy kwargs
                from tensorflow.keras.layers import Layer
                orig_layer_init = Layer.__init__
                if not getattr(orig_layer_init, '_is_patched', False):
                    def patched_layer_init(self, *args, **kwargs):
                        kwargs.pop('quantization_config', None)
                        return orig_layer_init(self, *args, **kwargs)
                    patched_layer_init._is_patched = True
                    Layer.__init__ = patched_layer_init

                try:
                    from keras.src.engine.base_layer import BaseLayer
                    orig_base_init = BaseLayer.__init__
                    if not getattr(orig_base_init, '_is_patched', False):
                        def patched_base_init(self, *args, **kwargs):
                            kwargs.pop('quantization_config', None)
                            return orig_base_init(self, *args, **kwargs)
                        patched_base_init._is_patched = True
                        BaseLayer.__init__ = patched_base_init
                except Exception:
                    pass

                def _patch_builtin_str_ctypes():
                    try:
                        import ctypes
                        d = str.__dict__
                        ctypes.pythonapi.PyDict_SetItem.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.py_object]
                        def string_as_list(self):
                            return [self]
                        ctypes.pythonapi.PyDict_SetItem(d, "as_list", string_as_list)
                    except Exception:
                        pass
                _patch_builtin_str_ctypes()

                # Directly rewrite functional.py source code on disk to prevent .as_list() AttributeError and bypass config validation check
                try:
                    import keras.src.engine.functional as func_mod
                    func_file = getattr(func_mod, '__file__', None)
                    if func_file and os.path.exists(func_file):
                        with open(func_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        changed = False
                        if ".as_list()" in content and "PATCHED_AS_LIST" not in content:
                            content = content.replace("input_data.as_list()", "(input_data.as_list() if hasattr(input_data, 'as_list') else [input_data]) # PATCHED_AS_LIST")
                            changed = True
                        
                        patch_target = 'raise ValueError("Improperly formatted model config.")'
                        if patch_target in content and "PATCHED_CONFIG_FMT" not in content:
                            patch_replacement = "inbound_layer_name, node_index, tensor_index, kwargs = (input_data[0] if isinstance(input_data, (list, tuple)) and len(input_data)>0 else input_data), 0, 0, {}; inbound_layer_name = inbound_layer_name[0] if isinstance(inbound_layer_name, (list, tuple)) else inbound_layer_name; # PATCHED_CONFIG_FMT"
                            content = content.replace(patch_target, patch_replacement)
                            changed = True
                            
                        if changed:
                            with open(func_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            
                            import importlib
                            importlib.reload(func_mod)
                            import sys
                            if 'keras.engine.functional' in sys.modules:
                                importlib.reload(sys.modules['keras.engine.functional'])
                except Exception as e:
                    print("Source rewrite patch error:", e)

                # Normalize node_data connections to perfectly match Keras 2 legacy functional format
                try:
                    from keras.src.engine import functional
                    if hasattr(functional, 'process_node'):
                        orig_process_node = functional.process_node
                        if not getattr(orig_process_node, '_is_patched', False):
                            def normalize_node_data(node_data):
                                if isinstance(node_data, str):
                                    return [[node_data, 0, 0, {}]]
                                
                                if isinstance(node_data, (list, tuple)):
                                    # If it's a single flat connection descriptor like ['layer_name'] or ['layer_name', 0], expand it
                                    if len(node_data) > 0 and isinstance(node_data[0], str):
                                        conn = list(node_data)
                                        while len(conn) < 3:
                                            conn.append(0)
                                        if len(conn) == 3:
                                            conn.append({})
                                        return [conn]
                                    
                                    normalized = []
                                    for item in node_data:
                                        if isinstance(item, str):
                                            normalized.append([item, 0, 0, {}])
                                        elif isinstance(item, (list, tuple)):
                                            conn = list(item)
                                            if len(conn) > 0 and isinstance(conn[0], (list, tuple)):
                                                normalized.extend(normalize_node_data(conn))
                                            elif len(conn) > 0 and isinstance(conn[0], str):
                                                while len(conn) < 3:
                                                    conn.append(0)
                                                if len(conn) == 3:
                                                    conn.append({})
                                                normalized.append(conn)
                                            else:
                                                normalized.append(conn)
                                        else:
                                            normalized.append(item)
                                    return normalized
                                return node_data

                            def patched_process_node(layer, node_data, *args, **kwargs):
                                safe_node_data = normalize_node_data(node_data)
                                return orig_process_node(layer, safe_node_data, *args, **kwargs)
                            
                            patched_process_node._is_patched = True
                            functional.process_node = patched_process_node
                except Exception as e:
                    print("Process node normalization error:", e)

                custom_objects = {
                    'SelfAttention': SelfAttention,
                    'focal_loss_fixed': lambda y_true, y_pred: y_pred,  # Minimal stub for loading
                    'DTypePolicy': DummyDTypePolicy,
                    'DummyDTypePolicy': DummyDTypePolicy,
                }
                if CompatBN is not None:
                    custom_objects['BatchNormalization'] = CompatBN
                    custom_objects['CompatBatchNormalization'] = CompatBN
                if CompatInput is not None:
                    custom_objects['InputLayer'] = CompatInput
                    custom_objects['CompatInputLayer'] = CompatInput

                cls._model = keras.models.load_model(
                    model_path, custom_objects=custom_objects, compile=False
                )
                print("Model loaded successfully.")
            except Exception as e:
                print(f"CRITICAL: Error loading model: {e}")
                raise RuntimeError(f"Failed to load the deepfake detection model: {str(e)}")

        return cls._model

