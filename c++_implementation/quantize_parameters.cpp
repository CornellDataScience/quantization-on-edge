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

std::tuple<std::vector<int>, float, int> linear_quantize_data(const std::vector<float>& data, int bit_size) {
    int qmin = 0;
    int qmax = (1 << bit_size) - 1;
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    float scale = (max_val - min_val) / (qmax - qmin);
    if (scale == 0) {
        scale = 1.0f;
    }

    int zero_point = static_cast<int>(std::round(qmin - min_val / scale));
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
        std::vector<float> param_array;

        // Check if the JSON value is an array of arrays or a flat array of numbers.
        if (item.value().is_array() && !item.value().empty() && item.value()[0].is_array()) {
            // Flatten the nested array.
            for (const auto& inner : item.value()) {
                for (const auto& num : inner) {
                    param_array.push_back(num.get<float>());
                }
            }
        } else {
            // It is already a flat array.
            for (const auto& num : item.value()) {
                param_array.push_back(num.get<float>());
            }
        }

        std::tuple<std::vector<int>, float, int> quantResult = linear_quantize_data(param_array, bit_size);
        std::vector<int> Q = std::get<0>(quantResult);
        float scale = std::get<1>(quantResult);
        int zero_point = std::get<2>(quantResult);

        quantized_params[param_name] = {
            {"quantized", Q},
            {"scale", scale},
            {"zero_point", zero_point},
            {"bit_width", bit_size}
        };
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
    std::string input_json = "unquantized_params.json";
    std::string output_json = "quantized_params.json";
    int bit_size = 8;

    quantize_parameters(input_json, output_json, bit_size);
    std::cout << "Quantized " << input_json << " -> " << output_json << " (" << bit_size << "-bit)" << std::endl;
    return 0;
}