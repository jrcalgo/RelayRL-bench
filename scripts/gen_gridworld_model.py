#!/usr/bin/env python3
"""
Generate a random initial ONNX policy model for the GridWorld environment.

Usage:
    python3 scripts/gen_gridworld_model.py [--obs-dim N] [--act-dim M] [--output-dir DIR]

Requirements:
    pip install onnx numpy

The generated model is a 2-layer MLP (obs_dim → 64 → act_dim) with ReLU activation.
Weights are initialized randomly (seed=42 for reproducibility).
"""

import argparse
import json
import os

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not found. Install with: pip install onnx numpy")
    raise SystemExit(1)


def generate_model(obs_dim: int, act_dim: int, output_dir: str, seed: int = 42):
    np.random.seed(seed)
    scale = 0.1

    W1 = (np.random.randn(obs_dim, 64) * scale).astype(np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    W2 = (np.random.randn(64, act_dim) * scale).astype(np.float32)
    b2 = np.zeros(act_dim, dtype=np.float32)

    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["h1"]),
        helper.make_node("Add",    ["h1", "b1"],    ["h1b"]),
        helper.make_node("Relu",   ["h1b"],          ["h1r"]),
        helper.make_node("MatMul", ["h1r", "W2"],   ["h2"]),
        helper.make_node("Add",    ["h2", "b2"],    ["output"]),
    ]

    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
    ]

    input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [None, obs_dim])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, act_dim])

    graph = helper.make_graph(nodes, "gridworld_policy", [input_vi], [output_vi],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    os.makedirs(output_dir, exist_ok=True)
    model_file = "gridworld_policy.onnx"
    onnx.save(model, os.path.join(output_dir, model_file))

    metadata = {
        "model_file": model_file,
        "model_type": "onnx",
        "input_dtype": {"NdArray": "F32"},
        "output_dtype": {"NdArray": "F32"},
        "input_shape": [1, obs_dim],
        "output_shape": [1, act_dim],
        "default_device": "cpu",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved to {output_dir}/")
    print(f"  input_shape:  [batch, {obs_dim}]")
    print(f"  output_shape: [batch, {act_dim}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GridWorld ONNX policy model")
    parser.add_argument("--obs-dim",    type=int, default=100,    help="Observation dimension (default: 100 for 10x10 grid)")
    parser.add_argument("--act-dim",    type=int, default=4,      help="Action dimension (default: 4)")
    parser.add_argument("--output-dir", type=str, default="model", help="Output directory (default: model/)")
    parser.add_argument("--seed",       type=int, default=42,     help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_model(args.obs_dim, args.act_dim, args.output_dir, args.seed)
