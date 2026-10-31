"""
TFLite (float32 + full-int8) and ONNX exports of the MLP controller for the MCU benchmark, built
from the numpy weights saved by export_rev_mcu_package.py (no torch needed; run in an env with
tensorflow and onnx):

  <tfenv>/bin/python scripts/export_rev_mcu_tflite.py --pkg data/revision/mcu_package \
      --windows data/revision/distill/RT4_p202/windows.npz

Writes mlp_f32.tflite, mlp_int8.tflite, mlp_f32.onnx and tflite_info.json (decision agreement of
each file with the float32 reference on the 100 test windows, int8 input/output quantisation).
"""
import argparse
import json
import os

import numpy as np


def heads(q):
    return np.stack([q[..., 3 * j:3 * j + 3].argmax(-1) for j in range(3)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', required=True)
    ap.add_argument('--windows', required=True)
    args = ap.parse_args()
    r = np.load(os.path.join(args.pkg, 'mlp_reference.npz'))
    W1, b1, W2, b2, W3, b3 = (r[k] for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))
    x_test, y_ref = r['x'].astype(np.float32), r['y_f32']
    xa = np.load(args.windows)['X'].astype(np.float32).mean(1)        # 10,000 windows for calibration
    info = {}

    import tensorflow as tf
    m = tf.keras.Sequential([tf.keras.Input((80,)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(9)])
    m.layers[0].set_weights([W1.T, b1]); m.layers[1].set_weights([W2.T, b2]); m.layers[2].set_weights([W3.T, b3])
    y_keras = m.predict(x_test, verbose=0)
    info['keras_vs_reference_max_abs_diff'] = float(np.abs(y_keras - y_ref).max())

    def run_tflite(path, x):
        it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        out = []
        for v in x:
            xin = v[None].astype(np.float32)
            if i['dtype'] == np.int8:
                s, z = i['quantization']; xin = np.clip(np.round(xin / s + z), -128, 127).astype(np.int8)
            it.set_tensor(i['index'], xin); it.invoke(); y = it.get_tensor(o['index'])[0]
            if o['dtype'] == np.int8:
                s, z = o['quantization']; y = (y.astype(np.float32) - z) * s
            out.append(y)
        return np.stack(out), i, o

    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    f32 = conv.convert(); open(os.path.join(args.pkg, 'mlp_f32.tflite'), 'wb').write(f32)
    y, _, _ = run_tflite(os.path.join(args.pkg, 'mlp_f32.tflite'), x_test)
    info['tflite_f32'] = {'bytes': len(f32), 'decision_agreement': float((heads(y) == heads(y_ref)).all(1).mean()), 'max_abs_diff': float(np.abs(y - y_ref).max())}

    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([xa[i:i + 1]] for i in range(0, len(xa), 10))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
    i8 = conv.convert(); open(os.path.join(args.pkg, 'mlp_int8.tflite'), 'wb').write(i8)
    y, i, o = run_tflite(os.path.join(args.pkg, 'mlp_int8.tflite'), x_test)
    info['tflite_int8'] = {'bytes': len(i8), 'decision_agreement': float((heads(y) == heads(y_ref)).all(1).mean()), 'max_abs_diff': float(np.abs(y - y_ref).max()),
                           'input_scale_zero_point': [float(i['quantization'][0]), int(i['quantization'][1])], 'output_scale_zero_point': [float(o['quantization'][0]), int(o['quantization'][1])]}

    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        inits = [numpy_helper.from_array(a.astype(np.float32), n) for a, n in ((W1, 'W1'), (b1, 'b1'), (W2, 'W2'), (b2, 'b2'), (W3, 'W3'), (b3, 'b3'))]
        nodes = [helper.make_node('Gemm', ['x', 'W1', 'b1'], ['a1'], transB=1), helper.make_node('Relu', ['a1'], ['h1']),
                 helper.make_node('Gemm', ['h1', 'W2', 'b2'], ['a2'], transB=1), helper.make_node('Relu', ['a2'], ['h2']),
                 helper.make_node('Gemm', ['h2', 'W3', 'b3'], ['y'], transB=1)]
        g = helper.make_graph(nodes, 'mlp_controller', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 80])], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 9])], inits)
        model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); model.ir_version = 8
        onnx.checker.check_model(model); onnx.save(model, os.path.join(args.pkg, 'mlp_f32.onnx')); info['onnx'] = 'mlp_f32.onnx (opset 13)'
    except Exception as e:
        info['onnx'] = f'skipped: {e}'
    json.dump(info, open(os.path.join(args.pkg, 'tflite_info.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
