//! Minimal hand-rolled ONNX protobuf encoder.
//!
//! Builds a valid `ModelProto` byte string for a fully-connected MLP with ReLU
//! activations between hidden layers (but NOT on the output layer). No external
//! protobuf crate is needed — the encoding follows the wire format described in
//! the Protocol Buffers encoding specification and the ONNX IR spec (opset 17).
//!
//! The resulting bytes can be loaded directly into ORT via
//! `Session::builder()?.commit_from_memory(&bytes)`, avoiding any filesystem
//! round-trip when updating a trained model.

// ── Protobuf wire-format helpers ────────────────────────────────────────────

/// Encode a 64-bit value as a varint, appending to `buf`.
fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Append a varint field (wire type 0).
fn field_varint(field: u32, v: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint(&mut buf, ((field as u64) << 3) | 0);
    write_varint(&mut buf, v);
    buf
}

/// Append a signed i64 as varint (sign-extended as u64).
fn field_i64(field: u32, v: i64) -> Vec<u8> {
    field_varint(field, v as u64)
}

/// Append an i32 as varint.
fn field_i32(field: u32, v: i32) -> Vec<u8> {
    field_varint(field, v as u32 as u64)
}

/// Append a length-delimited field (wire type 2).
fn field_bytes(field: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint(&mut buf, ((field as u64) << 3) | 2);
    write_varint(&mut buf, data.len() as u64);
    buf.extend_from_slice(data);
    buf
}

/// Append a string field (wire type 2, UTF-8 bytes).
fn field_str(field: u32, s: &str) -> Vec<u8> {
    field_bytes(field, s.as_bytes())
}

/// Append an embedded message field (wire type 2).
fn field_msg(field: u32, msg: &[u8]) -> Vec<u8> {
    field_bytes(field, msg)
}

/// Append a 32-bit fixed field (wire type 5) for a float.
fn field_f32(field: u32, v: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    write_varint(&mut buf, ((field as u64) << 3) | 5);
    buf.extend_from_slice(&v.to_le_bytes());
    buf
}

// ── ONNX proto message builders ─────────────────────────────────────────────

/// `TensorProto` — an initializer (weight or bias).
/// `dims`: shape as int64 slice; `float_data`: row-major f32 values.
fn make_tensor_proto(name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    for &d in dims {
        buf.extend(field_i64(1, d)); // dims: repeated int64, field 1
    }
    buf.extend(field_i32(2, 1)); // data_type = FLOAT (1), field 2
    buf.extend(field_str(8, name)); // name: string, field 8
    // raw_data: little-endian f32 bytes, field 9
    let raw: Vec<u8> = float_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    buf.extend(field_bytes(9, &raw));
    buf
}

/// `AttributeProto` for an integer attribute (e.g., `transB = 1`).
fn make_attr_int(name: &str, value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(field_str(1, name)); // name, field 1
    buf.extend(field_i32(20, 2)); // type = INT (2), field 20
    buf.extend(field_i64(3, value)); // i: int64, field 3
    buf
}

/// `AttributeProto` for a float attribute (e.g., `alpha = 1.0`).
fn make_attr_float(name: &str, value: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(field_str(1, name)); // name, field 1
    buf.extend(field_i32(20, 1)); // type = FLOAT (1), field 20
    buf.extend(field_f32(2, value)); // f: float, field 2
    buf
}

/// `NodeProto` for a `Gemm` node: `Y = alpha * A @ B + beta * C`.
/// Burn Linear stores weights as `[in, out]`, so we use `transB=0` and export
/// weights directly as `[in_dim, out_dim]` matching Burn's row-major layout.
fn make_gemm_node(name: &str, inputs: &[&str], output: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    for &inp in inputs {
        buf.extend(field_str(1, inp)); // input: repeated string, field 1
    }
    buf.extend(field_str(2, output)); // output, field 2
    buf.extend(field_str(3, name)); // name, field 3
    buf.extend(field_str(4, "Gemm")); // op_type, field 4
    buf.extend(field_msg(5, &make_attr_int("transB", 0)));   // attribute, field 5
    buf.extend(field_msg(5, &make_attr_float("alpha", 1.0)));
    buf.extend(field_msg(5, &make_attr_float("beta", 1.0)));
    buf
}

/// `NodeProto` for a `Relu` activation.
fn make_relu_node(name: &str, input: &str, output: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(field_str(1, input));
    buf.extend(field_str(2, output));
    buf.extend(field_str(3, name));
    buf.extend(field_str(4, "Relu"));
    buf
}

/// A concrete dimension value (int64 for static dims).
fn dim_value(v: i64) -> Vec<u8> {
    field_i64(1, v) // dim_value, field 1
}

/// A symbolic (dynamic) dimension, e.g. "batch".
fn dim_param(s: &str) -> Vec<u8> {
    field_str(2, s) // dim_param, field 2
}

/// `TensorShapeProto` — a list of dimensions.
fn make_shape(dims: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for d in dims {
        buf.extend(field_msg(1, d)); // dim: repeated Dimension, field 1
    }
    buf
}

/// `TypeProto.Tensor` — elem_type (1=FLOAT) + shape.
fn make_type_tensor(elem_type: i32, shape: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(field_i32(1, elem_type)); // elem_type, field 1
    buf.extend(field_msg(2, shape)); // shape: TensorShapeProto, field 2
    buf
}

/// `TypeProto` — wraps a `TypeProto.Tensor`.
fn make_type(tensor_type: &[u8]) -> Vec<u8> {
    field_msg(1, tensor_type) // tensor_type, field 1
}

/// `ValueInfoProto` — named tensor with type and dynamic batch dim.
fn make_value_info(name: &str, elem_type: i32, dims: Vec<Vec<u8>>) -> Vec<u8> {
    let shape = make_shape(&dims);
    let type_tensor = make_type_tensor(elem_type, &shape);
    let type_proto = make_type(&type_tensor);
    let mut buf = Vec::new();
    buf.extend(field_str(1, name)); // name, field 1
    buf.extend(field_msg(2, &type_proto)); // type, field 2
    buf
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Build raw ONNX `ModelProto` bytes for a fully-connected MLP.
///
/// `layer_specs` is a slice of `(in_dim, out_dim, weights, biases)` where:
/// - `weights` is a row-major `f32` slice of shape `[in_dim, out_dim]` (Burn's Linear layout)
/// - `biases` is a `f32` slice of length `out_dim`
///
/// ReLU activations are inserted between all layers **except** the last.
///
/// The resulting bytes can be passed directly to ORT's `commit_from_memory`.
pub fn build_onnx_mlp_bytes(layer_specs: &[(usize, usize, Vec<f32>, Vec<f32>)]) -> Vec<u8> {
    let n = layer_specs.len();
    assert!(n > 0, "layer_specs must be non-empty");

    let obs_dim = layer_specs[0].0;
    let act_dim = layer_specs[n - 1].1;

    let mut nodes: Vec<Vec<u8>> = Vec::new();
    let mut initializers: Vec<Vec<u8>> = Vec::new();
    let mut current_input = "obs".to_string();

    for (i, (in_dim, out_dim, weights, biases)) in layer_specs.iter().enumerate() {
        let w_name = format!("W{i}");
        let b_name = format!("b{i}");
        let pre_name = format!("pre{i}");
        let relu_name = format!("h{i}");
        let is_last = i == n - 1;

        // Weight initializer: shape [in_dim, out_dim] matching Burn's row-major layout
        initializers.push(make_tensor_proto(
            &w_name,
            &[*in_dim as i64, *out_dim as i64],
            weights,
        ));
        // Bias initializer: shape [out_dim]
        initializers.push(make_tensor_proto(&b_name, &[*out_dim as i64], biases));

        let gemm_output = if is_last { "output".to_string() } else { pre_name.clone() };

        nodes.push(make_gemm_node(
            &format!("gemm{i}"),
            &[&current_input, &w_name, &b_name],
            &gemm_output,
        ));

        if !is_last {
            nodes.push(make_relu_node(&format!("relu{i}"), &pre_name, &relu_name));
            current_input = relu_name;
        }
    }

    // Input: obs [batch, obs_dim]
    let input_info = make_value_info(
        "obs",
        1,
        vec![dim_param("batch"), dim_value(obs_dim as i64)],
    );
    // Output: output [batch, act_dim]
    let output_info = make_value_info(
        "output",
        1,
        vec![dim_param("batch"), dim_value(act_dim as i64)],
    );

    // GraphProto
    let mut graph = Vec::new();
    for node in &nodes {
        graph.extend(field_msg(1, node)); // node: repeated NodeProto, field 1
    }
    graph.extend(field_str(2, "mlp")); // name, field 2
    for init in &initializers {
        graph.extend(field_msg(5, init)); // initializer: repeated TensorProto, field 5
    }
    graph.extend(field_msg(11, &input_info)); // input, field 11
    graph.extend(field_msg(12, &output_info)); // output, field 12

    // OperatorSetIdProto: opset domain="" version=17
    let mut opset = Vec::new();
    opset.extend(field_str(1, "")); // domain
    opset.extend(field_i64(2, 17)); // version

    // ModelProto
    let mut model = Vec::new();
    model.extend(field_i64(1, 7)); // ir_version = 7, field 1
    model.extend(field_msg(8, &opset)); // opset_import, field 8
    model.extend(field_msg(7, &graph)); // graph, field 7
    model
}
