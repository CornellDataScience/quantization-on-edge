#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>


// Function to compute scale and zero point values

std::pair<float, int> compute_s_z_values(float r_min, float r_max, int bit_size){
    if (r_min == r_max) {
        return {1.0f, 0};
    }


    float S = (r_max - r_min) / (std::pow(2, bit_size) - 1);
    int Z = std::round(-r_min / S);

    return {S, Z};
}

//Function to linearly quantize data
std::tuple<std::vector<int>, float, int> linear_quant_data(const std::vector<float>& data, int bit_size) {
    // find min and max vals in data
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float rmax = *max_it;
    float rmin = *min_it;

    //computer scale and zero point
    auto [S, Z] = compute_s_z_values(rmin, rmax, bit_size);

    //quant the data

    std::vector<int> Q(data.size());
    int max_val = std::pow(2, bit_size) - 1;

    for (size_t i = 0; i < data.size(); i++) {
        int q_val = std::round(data[i] / S + Z);

        //clip the vals
        Q[i] = std::max(0, std::min(q_val, max_val));
    }

    return {Q, S, Z};
}

//function to linearly dequantize data

std::vector<float> linear_dequantize_data(const std::vector<int>& data, float S, int Z) {
    std::vector<float> dequantized(data.size());

    for (size_t i = 0; i < data.size(); i++) {
        dequantized[i] = (data[i] - Z) * S;
    }

    return dequantized;
}

float compute_mse(const std::vector<float>& original, const std::vector<float>& reconstructed) {

    float sum_squared_diff = 0.0f;

    for (size_t i = 0; i < original.size(); i++) {
        float diff = original[i] - reconstructed[i];
        sum_squared_diff += diff * diff;
    }

    return sum_squared_diff / original.size();
}

int main() {
    //sample data
    std::vector<float> data = {-0.30f, -0.22f, -0.24f, 0.0f, 0.35f};

    int bit_size = 8;

    // quantize the data

    auto [Q, S, Z] = linear_quant_data(data, bit_size);

    //dequantize data

    std::vector<float> dQ = linear_dequantize_data(Q, S, Z);

    //print for checking results

    std::cout << "Original Data: " << std::endl;

    for (const auto& val: data) {
        std::cout << val << " ";

    }

    std::cout << std::endl;

    std::cout << "\nQuantized (" << bit_size << "-bit):" << std::endl;

    for (const auto& val : Q) {
        std::cout << val << " ";
    }

    std::cout << std::endl;

    std::cout << "\nDequantized:" << std::endl;

    for (const auto& val : dQ) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "\nScale (S:) " << S << ", Zero-Point (Z): " << Z << std::endl;
    std::cout << "Quantization error with MSE: " << compute_mse(data, dQ) << std::endl;

    return 0;

}
