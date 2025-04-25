import onnx
import json
import numpy as np
from onnx import numpy_helper, helper
from linear_quantization import linear_quantize_data

def prepare_qat_model(onnx_model_path: str, activation_params_path: str, output_qat_model_path: str, bit_size: int = 8):
    # Load the unquantized model and activation parameters
    model = onnx.load(onnx_model_path)
    graph = model.graph

    with open(activation_params_path, 'r') as f:
        act_params = json.load(f)

    # Quantize each weight initializer and add scale/zero-point initializers
    existing_inits = {init.name for init in graph.initializer}
    new_initializers = []
    for init in list(graph.initializer):
        arr = numpy_helper.to_array(init).astype(np.float32)

        if init.name.lower().endswith("bias"):
            continue

        Q, S_w, Z_w = linear_quantize_data(arr, bit_size)

        quant_init = numpy_helper.from_array(Q, init.name)
        init.CopyFrom(quant_init)

        scale_name = init.name + "_scale"
        zp_name    = init.name + "_zero_point"
        new_initializers.append(
            numpy_helper.from_array(np.array([S_w], dtype=np.float32), scale_name)
        )
        new_initializers.append(
            numpy_helper.from_array(np.array([Z_w], dtype=np.int8), zp_name)
        )
        existing_inits.update({scale_name, zp_name})
        print(f"Quantized weight '{init.name}': S={S_w:.3e}, Z={Z_w}, shape={Q.shape}")

    # Insert Quantize/Dequantize nodes around each MatMul
    qdq_nodes = []
    for node in graph.node:
        if node.op_type not in ("MatMul"):
            continue

        act_in = node.input[0]
        s_act = act_in + "_activation_scale"
        z_act = act_in + "_activation_zero_point"
        if s_act not in existing_inits:
            S_a = act_params[act_in]["scale"]
            Z_a = act_params[act_in]["zero_point"]
            new_initializers += [
                numpy_helper.from_array(np.array([S_a], dtype=np.float32), s_act),
                numpy_helper.from_array(np.array([Z_a], dtype=np.int8), z_act)
            ]
            existing_inits.update({s_act, z_act})
        q_act = act_in + "_quant"
        dq_act = act_in + "_dq"

        qdq_nodes.append(helper.make_node(
            "QuantizeLinear",
            inputs=[act_in, s_act, z_act],
            outputs=[q_act],
            name=act_in + "_Quant",
        ))
        qdq_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[q_act, s_act, z_act],
            outputs=[dq_act],
            name=act_in + "_Dequant",
        ))

        w_in = node.input[1]
        q_w = w_in + "_quant"
        dq_w = w_in + "_dq"
        s_w = w_in + "_scale"
        z_w = w_in + "_zero_point"

        qdq_nodes.append(helper.make_node(
            "QuantizeLinear",
            inputs=[w_in, s_w, z_w],
            outputs=[q_w],
            name=w_in + "_Quant",
        ))
        qdq_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[q_w, s_w, z_w],
            outputs=[dq_w],
            name=w_in + "_Dequant",
        ))

        node.input[0] = dq_act
        node.input[1] = dq_w

        print(f"Inserted Q/DQ around '{node.name}' ({node.op_type})")

    # Prepend all Q/DQ nodes so they execute before the original ops
    graph.node[:0] = qdq_nodes

    # Wrap the model output with Q/DQ
    out_name = graph.output[0].name
    s_out = out_name + "_activation_scale"
    z_out = out_name + "_activation_zero_point"
    if s_out not in existing_inits:
        S_o = act_params[out_name]["scale"]
        Z_o = act_params[out_name]["zero_point"]
        new_initializers += [
            numpy_helper.from_array(np.array([S_o], dtype=np.float32), s_out),
            numpy_helper.from_array(np.array([Z_o], dtype=np.int8), z_out)
        ]
        existing_inits.update({s_out, z_out})

    q_out = helper.make_node("QuantizeLinear", [out_name, s_out, z_out], [out_name + "_quant"], name=out_name + "_Quant")
    dq_out = helper.make_node("DequantizeLinear", [out_name + "_quant", s_out, z_out], [out_name], name=out_name + "_Dequant")
    graph.node.extend([q_out, dq_out])
    print(f"Wrapped final output '{out_name}' with Q/DQ")

    # Append all new initializers and save QAT model, ready for training
    graph.initializer.extend(new_initializers)
    onnx.save(model, output_qat_model_path)
    print(f"\nQAT model (ready for training) saved to: {output_qat_model_path}")


if __name__ == "__main__":
    onnx_model_path = "models/model.onnx"
    activation_params_path = "activations/prep_activations.json"
    output_qat_model_path = "models/qat_model.onnx"
    prepare_qat_model(onnx_model_path, activation_params_path, output_qat_model_path)