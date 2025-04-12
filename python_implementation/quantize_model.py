import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
import numpy as np
from onnx import numpy_helper, helper
from onnx.reference.op_run import OpRun

def quantize(onnx_model_path, quantized_params_path, output_model_path):
    '''
    Quantize ONNX model and save quantized model

    Input
    -----
    onnx_model_path: file path of ONNX model to quantize
    quantized_params_path: file path of quantized parameter, scalar, and zero point values used to quantize
    output_model_path: file path to save quantized model to
    
    Output
    -----
    Saves quantized model to output_model_path 
    '''
    # Load unquantized model
    model = onnx.load(onnx_model_path)
    graph = model.graph

    print('** Original nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
    
    # Load JSON
    with open(quantized_params_path) as f:
        params = json.load(f)

    # Iterate through params, replace with quantized from JSON
    new_initializers = []
    for tensor in graph.initializer:
        if tensor.name in params:
            # Retrieve quantization params from JSON
            quantized_data = np.array(params[tensor.name]["quantized"], dtype=np.int8)
            weight_scale = np.array([params[tensor.name]["weight_scale"]], dtype=np.float32)
            weight_zero_point = np.array([params[tensor.name]["weight_zero_point"]], dtype=np.int8)
            activation_scale = np.array([params[tensor.name]["activation_scale"]], dtype=np.float32) # Need to move these to become associated with nodes, not initializers
            activation_zero_point = np.array([params[tensor.name]["activation_zero_point"]], dtype=np.int8)

            # Convert to ONNX tensors
            quantized_initializer = numpy_helper.from_array(quantized_data, tensor.name)
            weight_scale_initializer = numpy_helper.from_array(weight_scale, f"{tensor.name}_weight_scale")
            weight_zero_point_initializer = numpy_helper.from_array(weight_zero_point, f"{tensor.name}_weight_zero_point")
            activation_scale_initializer = numpy_helper.from_array(activation_scale, f"{tensor.name}_activation_scale")
            activation_zero_point_initializer = numpy_helper.from_array(activation_zero_point, f"{tensor.name}_activation_zero_point")

            # Add quantized initializer and scale/zero-point tensors
            new_initializers.extend([quantized_initializer, weight_scale_initializer, weight_zero_point_initializer, 
                                     activation_scale_initializer, activation_zero_point_initializer])

            print(f"Quantized {tensor.name} with weight_scale={weight_scale[0]}, weight_zero_point={weight_zero_point[0]}, \
                  activation_scale={activation_scale[0]}, and activation_zero_point={activation_zero_point[0]}")

    # Remove original initializers and replace with new ones
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers) # Note: consider creating initializers for ReLU

    # Iterate through nodes, replace with custom quantization nodes
    added_nodes = []
    removed_nodes = []

    prev_node = None
    for i,node in enumerate(graph.node):
        if node.op_type == "MatMul":
            matmul_node = node
            add_node = graph.node[i + 1]

            x = matmul_node.input[0]
            W = matmul_node.input[1]
            b = add_node.input[1]
            s_x = prev_node.name + "_activation_scale"
            s_W = matmul_node.name + "_weight_scale"

            output = add_node.output[0]

            matmul_add_fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "SymmetricMatMulAddFusion", 
                                                     op_type='SymmetricMatMulAddFusion', 
                                                     inputs=[x, W, b, s_x, s_W], 
                                                     outputs=[output], 
                                                     domain="quantize")
            

            added_nodes.append(matmul_add_fused_node)

            removed_nodes.append(matmul_node)
            removed_nodes.append(add_node)

        prev_node = node

    for node in removed_nodes:
        graph.node.remove(node)
        
    for node in added_nodes:
        graph.node.append(node)

    print('** Quantized nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
        
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid('', 13), helper.make_opsetid('quantize', 1)])
            
    # Save quantized ONNX model
    onnx.save(model, output_model_path)
    print(f"Quantized model saved to {output_model_path}")


class SymmetricMatMulAddFusion(OpRun):
    '''
    Perform fused matrix multiplication and bias addition assuming a symmetric quantization scheme

    Input
    -----
    x: name of activation initializer for previous layer
    W: name of weights initializer for current layer
    b: name of bias initializer for current layer
    s_x: name of activation scalar initializer for previous layer
    s_W: name of weights scalar initializer for current layer

    Output
    -----
    Returns unquantized result of matrix multiplication and bias addition
    '''

    op_domain = "quantize"

    def _run(self, x, W, b, s_x, s_W):
        x = x.copy()
        W = W.copy()
        b = b.copy()
        return (s_W * s_x * np.matmul(W, x) + b) # currently returns float32; all final operations should be int8
                                                 # calibration model should be float32 * int8
                                                 # consider fusing with relu to avoid extra quantization/dequantization


if __name__ == "__main__":
    onnx_model_path = "models/model.onnx"
    quantized_params_path = "params/quantized_params.json"
    output_model_path = "models/quantized_model.onnx"
    quantize(onnx_model_path, quantized_params_path, output_model_path)