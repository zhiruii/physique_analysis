"""
Convert physique_classifier_3class.h5 → TF.js layers-model format.

Handles Keras 3 → TF.js/Keras 2 format differences:
  - DTypePolicy objects → "float32" strings
  - inbound_nodes (Keras 3 tensor-dict format → [[layer, node, tensor]] tuples)
  - batch_shape → batch_input_shape for InputLayer
  - module/registered_name fields stripped throughout
  - Unsupported layer config keys removed (renorm, synchronized, quantization_config, etc.)
  - Weight names prefixed with layer name; DepthwiseConv2D kernel → depthwise_kernel
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

HERE       = Path(__file__).parent
MODEL_PATH = HERE / "backend" / "models" / "physique_classifier_3class.h5"
OUT_DIR    = HERE / "frontend" / "public" / "model"

# ── Config transformers ───────────────────────────────────────────────────────

# Keys to drop from layer-level dicts (not the config sub-dict)
_LAYER_DROP = {'module', 'registered_name', 'build_config'}

# Keys to drop from any layer config dict
_CONFIG_DROP = {
    'renorm', 'renorm_clipping', 'renorm_momentum', 'synchronized',  # BatchNorm
    'quantization_config',   # Dense (Keras 3 QAT artifact)
    'optional',              # InputLayer
    'trainable',             # redundant in config (already at layer level)
}

def _flatten_dtype(v):
    """DTypePolicy object → plain dtype string."""
    if isinstance(v, dict) and v.get('class_name') == 'DTypePolicy':
        return v.get('config', {}).get('name', 'float32')
    return v

def _clean_nested(obj):
    """
    Recursively strip Keras 3 metadata from nested serialisation objects
    (initializers, regularizers, constraints, activations).
    Keeps only {class_name, config}; recurses into config values.
    """
    if isinstance(obj, list):
        return [_clean_nested(x) for x in obj]
    if not isinstance(obj, dict):
        return obj

    # DTypePolicy → string
    if obj.get('class_name') == 'DTypePolicy':
        return _flatten_dtype(obj)

    # Keras 3 nested object pattern → Keras 2 pattern
    if 'class_name' in obj and 'config' in obj:
        return {
            'class_name': obj['class_name'],
            'config': _clean_nested(obj['config']),
        }

    # Plain dict — recurse, drop Keras-3-only metadata keys
    return {k: _clean_nested(v) for k, v in obj.items()
            if k not in ('module', 'registered_name')}

def _transform_config(class_name: str, raw_config: dict) -> dict:
    """Clean a single layer's config for TF.js compatibility."""
    out = {}
    for k, v in raw_config.items():
        if k in _CONFIG_DROP:
            continue
        if k == 'dtype':
            out[k] = _flatten_dtype(v) if isinstance(v, dict) else v
        elif k == 'batch_shape' and class_name == 'InputLayer':
            out['batch_input_shape'] = v          # rename for TF.js
        else:
            out[k] = _clean_nested(v)
    return out

def _transform_inbound_nodes(raw):
    """
    Keras 3 inbound_nodes is a LIST of call-record dicts:
      [{"args": (tensor_dict,),         "kwargs": {}}]   single-input
      [{"args": ([t1, t2],),            "kwargs": {}}]   multi-input (Add etc.)
      []                                                  InputLayer

    TF.js format:
      single input  → [["layer_name", node_idx, tensor_idx]]
      multi input   → [[["l1", 0, 0], ["l2", 0, 0]]]
      no input      → []
    """
    if not raw:
        return []

    def _history(tensor_dict):
        return list(tensor_dict['config']['keras_history'])

    # Keras 3: list of call-record dicts
    if isinstance(raw, list) and isinstance(raw[0], dict) and 'args' in raw[0]:
        result = []
        for call_record in raw:
            args = call_record.get('args', [])
            if not args:
                continue
            first = args[0]
            if isinstance(first, (list, tuple)):
                # Multi-input merge layer
                result.append([_history(t) for t in first])
            elif isinstance(first, dict) and first.get('class_name') == '__keras_tensor__':
                # Single-input layer
                result.append(_history(first))
        return result

    # Already TF.js / Keras-2-style — pass through
    return raw

def _transform_layers(layers: list) -> list:
    out = []
    for lyr in layers:
        cn = lyr.get('class_name', '')
        new = {
            'class_name': cn,
            'config':     _transform_config(cn, lyr.get('config', {})),
            'inbound_nodes': _transform_inbound_nodes(lyr.get('inbound_nodes', [])),
            'name':       lyr.get('name', ''),
        }
        out.append(new)
    return out

def build_topology(model) -> dict:
    cfg = model.get_config()
    new_cfg = {
        'name':         cfg.get('name', 'model'),
        'trainable':    cfg.get('trainable', True),
        'dtype':        'float32',
        'layers':       _transform_layers(cfg.get('layers', [])),
        'input_layers':  cfg.get('input_layers', []),
        'output_layers': cfg.get('output_layers', []),
    }
    return {
        'class_name':    'Functional',
        'config':        new_cfg,
        'keras_version': tf.keras.__version__,
        'backend':       'tensorflow',
    }

# ── Weight extraction ─────────────────────────────────────────────────────────

def extract_weights(model):
    """
    Returns (descriptors, raw_bytes).
    Weight names are {layer.name}/{var_name}, with DepthwiseConv2D's
    'kernel' renamed to 'depthwise_kernel' to match TF.js expectations.
    """
    descs, bufs = [], []
    for layer in model.layers:
        for w in layer.weights:
            var_name = w.name  # e.g. 'kernel', 'gamma', 'bias'

            # TF.js DepthwiseConv2D expects 'depthwise_kernel', not 'kernel'
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D) and var_name == 'kernel':
                var_name = 'depthwise_kernel'

            full_name = f'{layer.name}/{var_name}'
            arr = w.numpy().astype(np.float32)
            bufs.append(arr.tobytes())
            descs.append({
                'name':  full_name,
                'shape': list(arr.shape),
                'dtype': 'float32',
            })
    return descs, b''.join(bufs)

# ── Main ──────────────────────────────────────────────────────────────────────

print(f'Loading {MODEL_PATH} ...')
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

print(f'\nConverting to {OUT_DIR} ...')
OUT_DIR.mkdir(parents=True, exist_ok=True)

descs, raw_bytes = extract_weights(model)
shard = 'group1-shard1of1.bin'
(OUT_DIR / shard).write_bytes(raw_bytes)
print(f'  {shard}  ({len(raw_bytes):,} bytes)')

model_json = {
    'format':          'layers-model',
    'generatedBy':     f'keras {tf.keras.__version__}',
    'convertedBy':     'convert_model.py (standalone Keras3->TF.js)',
    'modelTopology':   build_topology(model),
    'weightsManifest': [{'paths': [shard], 'weights': descs}],
}

json_path = OUT_DIR / 'model.json'
json_path.write_text(json.dumps(model_json), encoding='utf-8')
print(f'  model.json  ({json_path.stat().st_size:,} bytes, {len(descs)} tensors)')
print('\nDone.')
