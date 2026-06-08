#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

#include "../common.hpp"
#include "../space/bitwise.hpp"
#include "./fastscan_impl.hpp"

namespace symqg {

constexpr size_t kExRaBitQBits = 4;  // 4-bit quantization
constexpr size_t kExRaBitQMaxVal = (1 << kExRaBitQBits) - 1;  // 15 for 4-bit

/**
 * @brief Fast multi-bit quantization using ExRaBitQ algorithm
 * 
 * Optimizes quantization codes to maximize inner product with original vector.
 * Uses greedy enumeration to find near-optimal scaling factor for each dimension.
 * 
 * @param rotated_unit_data Rotated and normalized residual vector
 * @param residual_norm L2 norm of residual
 * @param dim Padded dimension
 * @param compact_code Output: B-bit quantization codes (1 byte per dimension)
 * @param factor Output: 3 factors for approximate distance computation
 */
inline void exrabitq_global_code(
    const float* rotated_unit_data,
    float residual_norm,
    size_t dim,
    uint8_t* compact_code,
    float* factor
) {
    assert(dim % 64 == 0);
    constexpr double eps = 1e-5;
    constexpr int n_enum = 10;  // enumeration window size

    // Find max value for scaling initialization
    double max_o = -1;
    for (size_t i = 0; i < dim; ++i) {
        if (rotated_unit_data[i] > max_o) {
            max_o = rotated_unit_data[i];
        }
    }

    // Initialize search bounds
    double t_start = (double)(kExRaBitQMaxVal / 3) / (max_o + eps);
    double t_end = (double)(kExRaBitQMaxVal + n_enum) / (max_o + eps);

    // Initial quantization with t_start
    std::vector<int> cur_o_bar(dim);
    double sqr_denominator = dim * 0.25;
    double numerator = 0;

    for (size_t i = 0; i < dim; ++i) {
        cur_o_bar[i] = (int)((double)t_start * rotated_unit_data[i] + eps);
        cur_o_bar[i] = std::min((int)kExRaBitQMaxVal, cur_o_bar[i]);
        sqr_denominator += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
        numerator += (cur_o_bar[i] + 0.5) * rotated_unit_data[i];
    }

    // Priority queue for greedy enumeration
    std::priority_queue<
        std::pair<double, size_t>,
        std::vector<std::pair<double, size_t>>,
        std::greater<std::pair<double, size_t>>>
        next_t;

    for (size_t i = 0; i < dim; ++i) {
        double next_threshold = (cur_o_bar[i] + 1) / (rotated_unit_data[i] + eps);
        if (rotated_unit_data[i] > eps) {
            next_t.push(std::make_pair(next_threshold, i));
        }
    }

    double max_ip = 0;
    double best_t = t_start;

    // Greedy search for best scaling factor
    while (!next_t.empty()) {
        double cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        next_t.pop();

        if (cur_t > t_end) break;

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += 2 * update_o_bar;
        numerator += rotated_unit_data[update_id];

        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            best_t = cur_t;
        }

        // Add next threshold if not at max value
        if (update_o_bar < (int)kExRaBitQMaxVal) {
            double t_next = (update_o_bar + 1) / (rotated_unit_data[update_id] + eps);
            if (t_next < t_end) {
                next_t.push(std::make_pair(t_next, update_id));
            }
        }
    }

    // Final quantization with best_t
    std::vector<int> o_bar(dim);
    sqr_denominator = dim * 0.25;
    numerator = 0;
    int code_sum = 0;

    for (size_t i = 0; i < dim; ++i) {
        o_bar[i] = (int)((double)best_t * rotated_unit_data[i] + eps);
        o_bar[i] = std::min((int)kExRaBitQMaxVal, std::max(0, o_bar[i]));
        sqr_denominator += o_bar[i] * o_bar[i] + o_bar[i];
        numerator += (o_bar[i] + 0.5) * rotated_unit_data[i];
        code_sum += o_bar[i];
    }

    // Store codes: 2 codes per byte (4-bit each)
    for (size_t i = 0; i < dim; ++i) {
        if (i % 2 == 0) {
            compact_code[i / 2] = (uint8_t)(o_bar[i] & 0x0F);
        } else {
            compact_code[i / 2] |= (uint8_t)((o_bar[i] & 0x0F) << 4);
        }
    }

    // Compute factors for distance approximation
    float fac_norm = 1.F / std::sqrt((float)dim);
    float x0 = 0.F;
    for (size_t i = 0; i < dim; ++i) {
        x0 += rotated_unit_data[i] * (float)o_bar[i] * fac_norm;
    }

    factor[0] = residual_norm * residual_norm;
    if (residual_norm == 0.F || std::abs(x0) <= 1e-12F) {
        factor[1] = 0.F;
        factor[2] = 0.F;
        return;
    }

    factor[1] = -2.F * residual_norm / x0 * fac_norm;
    factor[2] = factor[1] * (float)code_sum;
}

}  // namespace symqg
