import onnx
import json
import numpy as np
from onnx import numpy_helper, helper
import sys

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
    Saves prepared model to output_prep_model_path
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
    for tensor in graph.initializer:
        if tensor.name in params:
            if params[tensor.name]["to_quantize"]:
                # Retrieve quantization params from JSON
                scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
                zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.float32)
                quantized_data = scale * np.array(params[tensor.name]["data"], dtype=np.float32) # Use float32 to be compatible with ONNXRunTime v1.18 MatMul operator
                
                # Convert to ONNX tensors
                quantized_initializer = numpy_helper.from_array(quantized_data, tensor.name)

                # Replace original initializer with quantized initializer
                tensor.CopyFrom(quantized_initializer)

                print(f"Quantized {tensor.name} with scale={scale[0]}, zero_point={zero_point[0]}")

    print('** Prepped nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
        
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            
    # Save quantized ONNX model
    onnx.save(model, output_prep_model_path)
    print(f"Prep model saved to {output_prep_model_path}")

def quantize(prep_model_path, quantized_params_path, quantized_activations_path, quantized_biases_path, output_model_path):
    '''
    Quantize ONNX model and save quantized model

    Input
    -----
    prep_model_path: file path of ONNX model to quantize
    quantized_params_path: file path of quantized parameter scalar and zero point values used to quantize
    quantized_activations_path: file path of scalar and zero point values for prepped model activations
    quantized_biases_path: file path of quantized bias, scalar, and zero point values used to quantize
    output_model_path: file path to save quantized model to
    
    Output
    -----
    Saves quantized model to output_model_path 
    '''
    # Load unquantized model
    model = onnx.load(prep_model_path)
    graph = model.graph

    print('** Original nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
        
    new_initializers = []
        
    # Load params JSON
    with open(quantized_params_path) as f:
        params = json.load(f)

    # Iterate through params, add scale and zero point
    for tensor in graph.initializer:
        if tensor.name in params:
            if params[tensor.name]["to_quantize"]: # Only quantize weights, not biases
                # Retrieve quantization params from JSON
                quantized_weight = np.array(params[tensor.name]["data"], dtype=np.int8)
                scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
                zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.int8)

                # Convert to ONNX tensors
                quantized_weight_initializer = numpy_helper.from_array(quantized_weight, tensor.name)
                quant_scale = numpy_helper.from_array(scale, tensor.name + "_scale")
                quant_zero_point = numpy_helper.from_array(zero_point, tensor.name + "_zero_point")

                # Replace original weight with quantized weight
                tensor.CopyFrom(quantized_weight_initializer)

                # Add quantized initializer and scale/zero-point tensors
                new_initializers.extend([quant_scale, quant_zero_point])

                print(f"Added {tensor.name} scale={scale[0]} and zero_point={zero_point[0]}")

    # Load biases JSON
    with open(quantized_biases_path) as f:
            biases = json.load(f) 

    for tensor in graph.initializer:
        if tensor.name in biases:
            # Retrieve quantizated bias from JSON
            quantized_bias = np.array(biases[tensor.name]["data"], dtype=np.int32)
            scale = np.array([biases[tensor.name]["scale"]], dtype=np.float32)

            # Convert to ONNX tensors
            quantized_biases_initializer = numpy_helper.from_array(quantized_bias, tensor.name)

            # Replace original biases with quantized biases
            tensor.CopyFrom(quantized_biases_initializer)

            print(f"Quantized {tensor.name} with scale={scale[0]}")

    # Iterate through nodes, replace with custom quantization nodes
    added_nodes = []
    removed_nodes = []

    activation_initializers = set()

    graph_nodes = graph.node

    # Add node to quantize model input
    quantize_node_name = "QuantizeLayer"
    input_name = graph.input[0].name
    s_x = quantize_node_name + "_activation_scale"
    Z = quantize_node_name + "_activation_zero_point"
    output_name = "quantized_input"
    quantize_node = helper.make_node(name=quantize_node_name, 
                                    op_type="Quantize", 
                                    inputs=[input_name, s_x, Z], 
                                    outputs=[output_name], 
                                    domain="ai.onnx.contrib")
    
    added_nodes.append(quantize_node)
    activation_initializers.add(s_x)
    activation_initializers.add(Z)

    prev_node = quantize_node
    
    for i,node in enumerate(graph_nodes):
        print(node.name)
        print([node.name for node in added_nodes])
        print([node.name for node in removed_nodes])
        print()
        if i == 0:
            input_name = "quantized_input"
            shape_initializer = node.input[1]
            s_x = node.name + "_activation_scale"
            Z = node.name + "_activation_zero_point"
            output_name = node.output[0]
            new_head = helper.make_node(name=node.name, 
                                        op_type=node.op_type, 
                                        inputs=[input_name, shape_initializer], 
                                        outputs=[output_name], 
                                        domain=node.domain)
            
            added_nodes.append(new_head)
            removed_nodes.append(node)

            activation_initializers.add(s_x)
            activation_initializers.add(Z)
            
        if node.op_type == "MatMul":
            matmul_node = node
            add_node = graph_nodes[i + 1]

            x = matmul_node.input[0]
            W = matmul_node.input[1]
            b = add_node.input[1]
            s_x = prev_node.name + "_activation_scale"
            s_W = W + "_scale"

            if i < len(graph_nodes) - 2:
                relu_node = graph_nodes[i + 2]

                s_R = relu_node.name + "_activation_scale"
                output = relu_node.output[0]
                
                matmul_add_relu_fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "SymmMatMulAddReLUFusion", 
                                                        op_type="SymmMatMulAddReLUFusion", 
                                                        inputs=[x, W, b, s_x, s_W, s_R], 
                                                        outputs=[output], 
                                                        domain="ai.onnx.contrib")
                added_nodes.append(matmul_add_relu_fused_node)
                removed_nodes.append(relu_node)

                activation_initializers.add(s_R)
            else: # Node is the last matmul node before output
                s_b = add_node.name + "_activation_scale"
                output = "quantized_output"

                matmul_add_fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "SymmMatMulAddFusion", 
                                                        op_type="SymmMatMulAddFusion", 
                                                        inputs=[x, W, b, s_x, s_W, s_b], 
                                                        outputs=[output], 
                                                        domain="ai.onnx.contrib")
                added_nodes.append(matmul_add_fused_node)
                
            removed_nodes.append(matmul_node)
            removed_nodes.append(add_node)

            activation_initializers.add(s_x)

        if node.op_type != "Reshape":
            prev_node = node

    # Add node to dequantize model output
    input_name = "quantized_output"
    s_x = prev_node.name + "_activation_scale"
    Z = prev_node.name + "_activation_zero_point"
    output_name = "output"
    dequantize_node = helper.make_node(name="DequantizeLayer", 
                                    op_type="Dequantize", 
                                    inputs=[input_name, s_x, Z], 
                                    outputs=[output_name], 
                                    domain="ai.onnx.contrib")
    added_nodes.append(dequantize_node)
    activation_initializers.add(s_x)
    activation_initializers.add(Z)

    # Replace nodes in graph
    for node in removed_nodes:
        if node in graph.node:
            graph.node.remove(node)
        
    for node in added_nodes:
        if node not in graph.node:
            graph.node.append(node)

    # Load activations JSON
    with open(quantized_activations_path) as f:
        activations = json.load(f)
    
    # Iterate through activations, add scale and zero point
    for node_name, _ in activations.items(): # activations is 2D dictionary
        for attribute, value in activations[node_name].items():
            if node_name + "_activation_" + attribute in activation_initializers:
                if attribute == "scale":
                    value = np.array([value], dtype=np.float32)
                elif attribute == "zero_point":
                    value = np.array([value], dtype=np.int8)

                attribute_initializer = numpy_helper.from_array(value, f"{node_name}_activation_{attribute}")

                new_initializers.extend([attribute_initializer])

                print(f"Added {node_name} {attribute}={value}")
        
    graph.initializer.extend(new_initializers)

    print('** Quantized nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))
        
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid('', 13), helper.make_opsetid("quantize", 1)])
            
    # Save quantized ONNX model
    onnx.save(model, output_model_path)
    print(f"Quantized model saved to {output_model_path}")


def quantize_asymmetric(prep_model_path, quantized_params_path, quantized_activations_path, quantized_biases_path, output_model_path):
    # Load unquantized model
    model = onnx.load(prep_model_path)
    graph = model.graph

    print('** Original nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))

    new_initializers = []

    # Load params JSON
    with open(quantized_params_path) as f:
        params = json.load(f)

    for tensor in graph.initializer:
        if tensor.name in params:
            if params[tensor.name]["to_quantize"]:
                quantized_weight = np.array(params[tensor.name]["data"], dtype=np.int8)
                scale = np.array([params[tensor.name]["scale"]], dtype=np.float32)
                zero_point = np.array([params[tensor.name]["zero_point"]], dtype=np.int8)

                quantized_weight_initializer = numpy_helper.from_array(quantized_weight, tensor.name)
                quant_scale = numpy_helper.from_array(scale, tensor.name + "_scale")
                quant_zero_point = numpy_helper.from_array(zero_point, tensor.name + "_zero_point")

                tensor.CopyFrom(quantized_weight_initializer)
                new_initializers.extend([quant_scale, quant_zero_point])
                print(f"Added {tensor.name} scale={scale[0]} and zero_point={zero_point[0]}")

    # Load biases JSON
    with open(quantized_biases_path) as f:
        biases = json.load(f)

    for tensor in graph.initializer:
        if tensor.name in biases:
            quantized_bias = np.array(biases[tensor.name]["data"], dtype=np.int32)
            scale = np.array([biases[tensor.name]["scale"]], dtype=np.float32)
            quantized_biases_initializer = numpy_helper.from_array(quantized_bias, tensor.name)
            tensor.CopyFrom(quantized_biases_initializer)
            print(f"Quantized {tensor.name} with scale={scale[0]}")

    added_nodes = []
    removed_nodes = []
    activation_initializers = set()
    graph_nodes = graph.node

    # Add node to quantize model input
    quantize_node_name = "QuantizeLayer"
    input_name = graph.input[0].name
    s_x = quantize_node_name + "_activation_scale"
    Z = quantize_node_name + "_activation_zero_point"
    output_name = "quantized_input"
    quantize_node = helper.make_node(name=quantize_node_name,
                                     op_type="Quantize",
                                     inputs=[input_name, s_x, Z],
                                     outputs=[output_name],
                                     domain="ai.onnx.contrib")

    added_nodes.append(quantize_node)
    activation_initializers.add(s_x)
    activation_initializers.add(Z)

    prev_node = quantize_node

    for i, node in enumerate(graph_nodes):
        if node.op_type == "MatMul":
            matmul_node = node
            add_node = graph_nodes[i + 1]

            x = matmul_node.input[0]
            W = matmul_node.input[1]
            b = add_node.input[1]
            s_x = prev_node.name + "_activation_scale"
            s_W = W + "_scale"

            # ADDED ZERO POINTS
            z_x = prev_node.name + "_activation_zero_point"
            z_W = W + "_zero_point"


            if i < len(graph_nodes) - 2:
                relu_node = graph_nodes[i + 2]
                s_R = relu_node.name + "_activation_scale"

                # ADDED OUTPUT ZERO POINT
                # z_R = relu_node.name + "_activation_zero_point"

                output = relu_node.output[0]

                # USE ASYMM OP AND EXTRA INPUTS
                fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "AsymmMatMulAddReLUFusion",
                                              op_type="AsymmMatMulAddReLUFusion",
                                              inputs=[x, W, b, s_x, s_W, s_R, z_x, z_W],
                                              outputs=[output],
                                              domain="ai.onnx.contrib")
                
                
                

                added_nodes.append(fused_node)
                removed_nodes.extend([matmul_node, add_node, relu_node])

                # activation_initializers.update([s_R, z_R])
                activation_initializers.update([s_R])
            else:
                s_b = add_node.name + "_activation_scale"

                # HANDLE OUTPUT ZERO POINT
                # z_b = add_node.name + "_activation_zero_point"

                output = "quantized_output"

                # USE ASYMM OP AND EXTRA INPUTS
                fused_node = helper.make_node(name=matmul_node.name[:matmul_node.name.rindex("/") + 1] + "AsymmMatMulAddFusion",
                                              op_type="AsymmMatMulAddFusion",
                                              inputs=[x, W, b, s_x, s_W, s_b, z_x, z_W],
                                              outputs=[output],
                                              domain="ai.onnx.contrib")
                # --- END ASYMMETRIC CHANGE ---

                added_nodes.append(fused_node)
                removed_nodes.extend([matmul_node, add_node])

            activation_initializers.update([s_x, z_x, z_W])

        if node.op_type != "Reshape":
            prev_node = node

    # Add node to dequantize model output
    input_name = "quantized_output"
    s_x = prev_node.name + "_activation_scale"
    Z = prev_node.name + "_activation_zero_point"
    output_name = "output"
    dequantize_node = helper.make_node(name="DequantizeLayer",
                                       op_type="Dequantize",
                                       inputs=[input_name, s_x, Z],
                                       outputs=[output_name],
                                       domain="ai.onnx.contrib")
    added_nodes.append(dequantize_node)
    activation_initializers.add(s_x)
    activation_initializers.add(Z)

    for node in removed_nodes:
        if node in graph.node:
            graph.node.remove(node)

    for node in added_nodes:
        if node not in graph.node:
            graph.node.append(node)

    # Load activations JSON
    with open(quantized_activations_path) as f:
        activations = json.load(f)

    for node_name, _ in activations.items():
        for attribute, value in activations[node_name].items():
            full_name = f"{node_name}_activation_{attribute}"
            if full_name in activation_initializers:
                if attribute == "scale":
                    value = np.array([value], dtype=np.float32)
                elif attribute == "zero_point":
                    value = np.array([value], dtype=np.int8)
                initializer = numpy_helper.from_array(value, full_name)
                new_initializers.append(initializer)
                print(f"Added {node_name} {attribute}={value}")

    graph.initializer.extend(new_initializers)

    print('** Quantized nodes **')
    for node in model.graph.node:
        print("name=%r type=%r input=%r output=%r" % (
            node.name, node.op_type, node.input, node.output))

    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid('', 13),
        helper.make_opsetid("quantize", 1)])

    onnx.save(model, output_model_path)
    print(f"Quantized model saved to {output_model_path}")




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected quantization mode argument.")
    else:
        mode = sys.argv[1]

        if mode == "prep":
            onnx_model_path = "models/model.onnx"
            quantized_params_path = "params/quantized_params.json"
            output_prep_model_path = "models/prep_model.onnx"
            prepare(onnx_model_path, quantized_params_path, output_prep_model_path)
        elif mode == "full":
            prep_model_path = "models/prep_model.onnx"
            quantized_params_path = "params/quantized_params.json"
            quantized_activations_path = "activations/quantized_activations.json"
            quantized_biases_path = "biases/quantized_biases.json"
            output_model_path = "models/quantized_model.onnx"
            quantize(prep_model_path, quantized_params_path, quantized_activations_path, quantized_biases_path, output_model_path)
            
            
        # ADDED THIS FOR ASYMMETRIC
        elif mode == "asymmetric":  
            prep_model_path = "models/prep_model.onnx"
            quantized_params_path = "params/quantized_params.json"
            quantized_activations_path = "activations/quantized_activations.json"
            quantized_biases_path = "biases/quantized_biases.json"
            output_model_path = "models/asymmetric_model.onnx"
            quantize_asymmetric(prep_model_path, quantized_params_path, quantized_activations_path, quantized_biases_path, output_model_path)
    
    
    
    
    
    
    

        
