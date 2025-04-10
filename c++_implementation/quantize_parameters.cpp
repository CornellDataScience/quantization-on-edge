#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <tuple>
#include <iomanip>
#include "json.hpp"

using json = nlohmann::json;

// Function for 1D quantization remains the same.
std::tuple<std::vector<int>, float, int> linear_quantize_data(const std::vector<float>& data, int bit_size) {
    int qmin = -128;
    int qmax = (1 << bit_size - 1) - 1;
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    float scale = (max_val - min_val) / (qmax - qmin);
    if (scale == 0) {
        scale = 1.0f;
    }

    int zero_point = 0;
    zero_point = std::max(qmin, std::min(zero_point, qmax));

    std::vector<int> Q;
    Q.reserve(data.size());
    for (float x : data) {
        int q = static_cast<int>(std::round(x / scale)) + zero_point;
        q = std::max(qmin, std::min(q, qmax));
        Q.push_back(q);
    }

    return std::make_tuple(Q, scale, zero_point);
}

void quantize_parameters(const std::string& input_path, const std::string& output_path, int bit_size) {
    std::ifstream inFile(input_path);
    if (!inFile) {
        std::cerr << "Error opening file: " << input_path << std::endl;
        return;
    }
    json params;
    inFile >> params;
    inFile.close();

    json quantized_params;
    for (auto& item : params.items()) {
        std::string param_name = item.key();
        auto &jparam = item.value();

        // If the parameter is a 2D array
        if (jparam.is_array() && !jparam.empty() && jparam[0].is_array()) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();

            // Compute overall min/max
            for (const auto &row : jparam) {
                for (const auto &val : row) {
                    float x = val.get<float>();
                    min_val = std::min(min_val, x);
                    max_val = std::max(max_val, x);
                }
            }
            int qmin = -128;
            int qmax = (1 << bit_size - 1) - 1;
            float scale = (max_val - min_val) / (qmax - qmin);
            if (scale == 0) {
                scale = 1.0f;
            }
            int zero_point = 0;
            zero_point = std::max(qmin, std::min(zero_point, qmax));

            // Quantize each row while preserving 2D structure
            json quantized_array = json::array();
            for (const auto &row : jparam) {
                json quant_row = json::array();
                for (const auto &val : row) {
                    float x = val.get<float>();
                    int q = static_cast<int>(std::round(x / scale)) + zero_point;
                    q = std::max(qmin, std::min(q, qmax));
                    quant_row.push_back(q);
                }
                quantized_array.push_back(quant_row);
            }

            quantized_params[param_name] = {
                {"bit_width", bit_size},
                {"quantized", quantized_array},
                {"scale", scale},
                {"zero_point", zero_point}
            };
        } else {
            // Otherwise, assume a 1D array and use the existing function.
            std::vector<float> data = jparam.get<std::vector<float>>();
            auto [Q, scale, zero_point] = linear_quantize_data(data, bit_size);
            quantized_params[param_name] = {
                {"bit_width", bit_size},
                {"quantized", Q},
                {"scale", scale},
                {"zero_point", zero_point}
            };
        }
    }

    std::ofstream outFile(output_path);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << output_path << std::endl;
        return;
    }
    outFile << std::setw(2) << quantized_params << std::endl;
    outFile.close();
}

int main() {
    std::string input_json = "../params/unquantized_params.json";
    std::string output_json = "../params/c++_quantized_params.json";
    int bit_size = 8;

    quantize_parameters(input_json, output_json, bit_size);
    std::cout << "Quantized " << input_json << " -> " << output_json << " (" << bit_size << "-bit)" << std::endl;
    return 0;
}
