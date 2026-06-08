#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "common.hpp"
#include "quantization/exrabitq.hpp"
#include "utils/rotator.hpp"

int main() {
    using namespace symqg;

    // Test parameters
    size_t dim = 128;  // Must be multiple of 64
    size_t padded_dim = 128;
    
    // Create test data
    std::vector<float> rotated_unit(padded_dim, 0.0f);
    
    // Initialize with some values
    for (size_t i = 0; i < dim; ++i) {
        rotated_unit[i] = (float)(i % 16) / 8.0f + 0.1f;  // Values in range [0.1, 2.0]
    }
    
    float residual_norm = 2.5f;
    
    // Allocate output buffers
    size_t code_bytes = padded_dim / 2;  // 4-bit: 2 codes per byte
    std::vector<uint8_t> compact_code(code_bytes, 0);
    std::vector<float> factors(3, 0.0f);
    
    std::cout << "Testing 4-bit ExRaBitQ Quantization\n";
    std::cout << "====================================\n";
    std::cout << "Dimension: " << dim << "\n";
    std::cout << "Padded Dimension: " << padded_dim << "\n";
    std::cout << "Code bytes: " << code_bytes << "\n";
    std::cout << "Residual norm: " << residual_norm << "\n\n";
    
    // Call quantization function
    exrabitq_global_code(
        rotated_unit.data(),
        residual_norm,
        padded_dim,
        compact_code.data(),
        factors.data()
    );
    
    // Verify output
    std::cout << "Output Factors:\n";
    std::cout << "  factor[0] (norm_sqr): " << std::fixed << std::setprecision(6) << factors[0] << "\n";
    std::cout << "  factor[1] (fac_dq):   " << std::fixed << std::setprecision(6) << factors[1] << "\n";
    std::cout << "  factor[2] (fac_vq):   " << std::fixed << std::setprecision(6) << factors[2] << "\n\n";
    
    // Display some codes
    std::cout << "First 16 codes (first 8 bytes):\n";
    for (size_t i = 0; i < 8 && i < code_bytes; ++i) {
        uint8_t byte = compact_code[i];
        uint8_t lo = byte & 0x0F;
        uint8_t hi = (byte >> 4) & 0x0F;
        std::cout << "  Byte " << i << ": lo=" << (int)lo << ", hi=" << (int)hi << "\n";
    }
    
    // Verify factor values
    std::cout << "\nVerification:\n";
    
    // factor[0] should be residual_norm^2
    float expected_factor0 = residual_norm * residual_norm;
    std::cout << "  factor[0] expected: " << expected_factor0 << ", got: " << factors[0];
    if (std::abs(factors[0] - expected_factor0) < 1e-5f) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
    }
    
    // Check if codes are in valid range [0, 15]
    bool codes_valid = true;
    for (size_t i = 0; i < code_bytes; ++i) {
        uint8_t lo = compact_code[i] & 0x0F;
        uint8_t hi = (compact_code[i] >> 4) & 0x0F;
        if (lo > 15 || hi > 15) {
            codes_valid = false;
            break;
        }
    }
    std::cout << "  All codes in range [0,15]: " << (codes_valid ? "✓" : "✗") << "\n";
    
    std::cout << "\n4-bit ExRaBitQ quantization test completed successfully!\n";
    
    return 0;
}
