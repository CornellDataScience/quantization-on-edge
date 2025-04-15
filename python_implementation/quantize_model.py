import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
import numpy as np
from onnx import numpy_helper, helper
from onnx.reference.op_run import OpRun

def prepare(onnx_model_path, quantized_params_path, output_prep_model_path):
    '''
    Prepare ONNX model for calibration

    Input
    -----
    onnx_model_path: file path of ONNX model to quantize
    quantized_params_path: file path of quantized parameter, scalar, and zero point values used to quantize
    output_prep_model_path: file path to save prepped model to
    
    Output
    -----
    Saves prepared model to output_model_path
    '''
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
            if params[tensor.name]["to_quantize"]:
                # Retrieve quantization params from JSON
                quantized_data = np.array(params[tensor.name]["data"], dtype=np.float32) # Use float32 to be compatible with ONNXRunTime v1.18 MatMul operator and multiplication with float32 activations
                scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
                zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.float32)

                # Convert to ONNX tensors
                quantized_initializer = numpy_helper.from_array(quantized_data, tensor.name)
                scale_initializer = numpy_helper.from_array(scale, f"{tensor.name}_scale")
                zero_point_initializer = numpy_helper.from_array(zero_point, f"{tensor.name}_zero_point")

                # Add quantized initializer and scale/zero-point tensors
                new_initializers.extend([quantized_initializer, scale_initializer, zero_point_initializer])

                print(f"Quantized {tensor.name} with scale={scale[0]}, zero_point={zero_point[0]}")
            else:
                new_initializers.extend([tensor])

    # Remove original initializers and replace with new ones
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers) # Note: consider creating initializers for ReLU

    print('** Prepped nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
        
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            
    # Save quantized ONNX model
    onnx.save(model, output_prep_model_path)
    print(f"Prep model saved to {output_prep_model_path}")

def quantize(onnx_model_path, quantized_activations_path, output_model_path):
    '''
    Quantize ONNX model and save quantized model

    Input
    -----
    onnx_model_path: file path of ONNX model to quantize
    quantized_activations_path: file path of scalar and zero point values for prepped model activations
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
    with open(quantized_activations_path) as f:
        activations = json.load(f)

    # Iterate through params, replace with quantized from JSON
    new_initializers = []
    for node_name, values in activations.items(): # activations is 2D dictionary
        scale = np.array([values["scale"]], dtype=np.float32)
        zero_point = np.array([values["zero_point"]], dtype=np.int8)

        scale_initializer = numpy_helper.from_array(scale, f"{node_name}_scale")
        zero_point_initializer = numpy_helper.from_array(zero_point, f"{node_name}_zero_point")
        
        new_initializers.extend([scale_initializer, zero_point_initializer])

        print(f"Quantized {node_name} with scale={scale[0]}, zero_point={zero_point[0]}")
        
    graph.initializer.extend(new_initializers)

    # Iterate through nodes, replace with custom quantization nodes
    added_nodes = []
    removed_nodes = []

    
    graph_nodes = graph.node

    # Add node to quantize model input
    input_name = graph.input[0].name
    s_x = input_name + "_activation_scalar"
    Z = input_name + "_activation_zero_point"
    output_name = "quantized_input"
    quantize_node = helper.make_node(name="QuantizeLayer", 
                                    op_type='Quantize', 
                                    inputs=[input_name, s_x, Z], 
                                    outputs=[output_name], 
                                    domain="quantize")
    
    added_nodes.append(quantize_node)

    prev_node = quantize_node
    
    for i,node in enumerate(graph_nodes):
        if node.op_type == "MatMul":
            matmul_node = node
            add_node = graph_nodes[i + 1]

            x = matmul_node.input[0]
            W = matmul_node.input[1]
            b = add_node.input[1]
            s_x = prev_node.name + "_activation_scale"
            s_W = matmul_node.name + "_scale"

            if i < len(graph_nodes) - 1:
                relu_node = graph_nodes[i + 2]

                s_R = relu_node.name + "_activation_scale"
                output = relu_node.output[0]
                
                matmul_add_relu_fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "SymmMatMulAddReLUFusion", 
                                                        op_type='SymmMatMulAddReLUFusion', 
                                                        inputs=[x, W, b, s_x, s_W, s_R], 
                                                        outputs=[output], 
                                                        domain="quantize")
                added_nodes.append(matmul_add_relu_fused_node)
                removed_nodes.append(relu_node)
            else: # Node is the original output node
                output = "quantized_output"

                matmul_add_fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "SymmMatMulAddFusion", 
                                                        op_type='SymmMatMulAddFusion', 
                                                        inputs=[x, W, b, s_x, s_W], 
                                                        outputs=[output], 
                                                        domain="quantize")
                added_nodes.append(matmul_add_fused_node)
                
            removed_nodes.append(matmul_node)
            removed_nodes.append(add_node)
                
        prev_node = node

    # Add node to dequantize model output
    input_name = "quantized_output"
    s_x = prev_node.name + "_activation_scalar"
    Z = prev_node.name + "_activation_zero_point"
    output_name = "output"
    dequantize_node = helper.make_node(name="DeuantizeLayer", 
                                    op_type='Dequantize', 
                                    inputs=[input_name, s_x, Z], 
                                    outputs=[output_name], 
                                    domain="quantize")
    added_nodes.append(dequantize_node)

    # Replace nodes in graph
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


class SymmMatMulAddFusion(OpRun):
    '''
    Perform fused matrix multiplication and bias addition assuming a symmetric quantization scheme

    Invariant: The output of this node is the final result of the graph

    Input
    -----
    x: name of activation initializer of previous layer
    W: name of weights initializer of current layer
    b: name of bias initializer of current layer
    s_x: name of activation scalar initializer of previous layer
    s_W: name of weights scalar initializer of current layer

    Output
    -----
    Returns float32 result of matrix multiplication and bias addition
    '''

    op_domain = "quantize"

    def _run(self, x, W, b, s_x, s_W):
        x = x.copy()
        W = W.copy()
        b = b.copy()
        return (s_W * s_x * (np.matmul(W, x) + b)) # FIXME need to make sure s_b = s_x * s_W for accuracy


class SymmMatMulAddReLUFusion(OpRun):
    '''
    Perform fused matrix multiplication, bias addition, and ReLU assuming a symmetric quantization scheme

    Input
    -----
    x: name of activation initializer of previous layer
    W: name of weights initializer of current layer
    b: name of bias initializer of current layer
    s_x: name of activation scalar initializer of previous layer
    s_W: name of weights scalar initializer of current layer
    s_R: name of ReLU scalar initializer of previous layer

    Output
    -----
    Returns unquantized result of matrix multiplication and bias addition
    '''

    op_domain = "quantize"

    def _run(self, x, W, b, s_x, s_W, s_R):
        x = x.copy()
        W = W.copy()
        b = b.copy()
        M = (s_x * s_W) / s_R
        return max(0, (np.matmul(W, x) + b) * M) # FIXME need to make sure s_b = s_x * s_W for accuracy


class Quantize(OpRun):
    '''
    Performs 8-bit linear quantization

    Input
    -----
    x: name of model input initializer
    s_x: name of scalar initializer of input
    Z: name of zero point initializer of input

    Output
    -----
    Returns quantized initializer
    '''
    op_domain = "quantize"
    
    def _run(self, x, s_x, Z):
        x = x.copy()
        bit_size = 8
        return np.clip(np.round(x / s_x + Z), -2**(bit_size-1), 2**(bit_size-1) - 1)
    
    
class Dequantize(OpRun):
    '''
    Performs 8-bit linear dequantization

    Input
    -----
    x: name of activation initializer of previous layer
    s_x: name of activation scalar initializer of previous layer
    Z: name of zero point initializer of previous layer

    Output
    -----
    Returns dequantized initializer
    '''
    op_domain = "quantize"

    def _run(self, x, s_x, Z):
        x = x.copy()
        return s_x * (x - Z)


if __name__ == "__main__":
    onnx_model_path = "models/model.onnx"
    quantized_params_path = "params/quantized_params.json"
    quantized_activations_path = "activations/quantized_activations.json"
    prep_model_path = "models/prep_model.onnx"
    output_model_path = "models/quantized_model.onnx"
    # quantize(onnx_model_path, quantized_params_path, output_model_path)
    prepare(onnx_model_path, quantized_params_path, prep_model_path)