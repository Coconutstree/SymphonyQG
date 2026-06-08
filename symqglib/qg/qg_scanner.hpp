#pragma once

#include <immintrin.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../quantization/fastscan_impl.hpp"

namespace symqg {

static inline void appro_dist_impl(
    size_t num_points,
    float query_norm_sqr,
    float query_norm,
    float width,
    float vl,
    const float* __restrict__ result,
    const float* __restrict__ data_norm_sqr,
    const float* __restrict__ fac_dq,
    const float* __restrict__ fac_vq,
    float* __restrict__ appro_dist
) {
#if defined(__AVX512F__)
    const __m512 query_norm_sqr_simd = _mm512_set1_ps(query_norm_sqr);
    const __m512 query_norm_simd = _mm512_set1_ps(query_norm);
    const __m512 width_simd = _mm512_set1_ps(width);
    const __m512 vl_simd = _mm512_set1_ps(vl);

    __m512 result_simd;
    __m512 data_norm_sqr_simd;
    __m512 fac_dq_simd;
    __m512 fac_vq_simd;

    for (size_t i = 0; i < num_points; i += 16) {
        result_simd = _mm512_loadu_ps(&result[i]);
        data_norm_sqr_simd = _mm512_loadu_ps(&data_norm_sqr[i]);
        fac_dq_simd = _mm512_loadu_ps(&fac_dq[i]);
        fac_vq_simd = _mm512_loadu_ps(&fac_vq[i]);

        data_norm_sqr_simd = _mm512_add_ps(data_norm_sqr_simd, query_norm_sqr_simd);

        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, width_simd);
        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, result_simd);

        fac_vq_simd = _mm512_fmadd_ps(fac_vq_simd, vl_simd, fac_dq_simd);
        fac_vq_simd = _mm512_mul_ps(fac_vq_simd, query_norm_simd);

        data_norm_sqr_simd = _mm512_add_ps(data_norm_sqr_simd, fac_vq_simd);
        _mm512_storeu_ps(&appro_dist[i], data_norm_sqr_simd);
    }
#elif defined(__AVX2__)
    const __m256 query_norm_sqr_simd = _mm256_set1_ps(query_norm_sqr);
    const __m256 query_norm_simd = _mm256_set1_ps(query_norm);
    const __m256 width_simd = _mm256_set1_ps(width);
    const __m256 vl_simd = _mm256_set1_ps(vl);

    __m256 result_simd;
    __m256 data_norm_sqr_simd;
    __m256 fac_dq_simd;
    __m256 fac_vq_simd;

    for (size_t i = 0; i < num_points; i += 8) {
        result_simd = _mm256_loadu_ps(&result[i]);
        data_norm_sqr_simd = _mm256_loadu_ps(&data_norm_sqr[i]);
        fac_dq_simd = _mm256_loadu_ps(&fac_dq[i]);
        fac_vq_simd = _mm256_loadu_ps(&fac_vq[i]);

        data_norm_sqr_simd = _mm256_add_ps(data_norm_sqr_simd, query_norm_sqr_simd);

        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, width_simd);
        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, result_simd);

        fac_vq_simd = _mm256_mul_ps(fac_vq_simd, vl_simd);
        fac_vq_simd = _mm256_add_ps(fac_vq_simd, fac_dq_simd);
        fac_vq_simd = _mm256_mul_ps(fac_vq_simd, query_norm_simd);

        data_norm_sqr_simd = _mm256_add_ps(data_norm_sqr_simd, fac_vq_simd);
        _mm256_storeu_ps(&appro_dist[i], data_norm_sqr_simd);
    }
    return;
#else
    std::cerr << "SIMD (AVX512 or AVX2) REQUIRED!\n";
    abort();
#endif
}

class QGScanner {
   private:
    // func for packing lookup tables
    size_t padded_dim_;
    size_t degree_bound_;

   public:
    QGScanner() = default;

    explicit QGScanner(size_t padded_dim, size_t degree_bound)
        : padded_dim_(padded_dim), degree_bound_(degree_bound) {}

    void pack_lut(const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT) const {
        pack_lut_impl(padded_dim_, byte_query, LUT);
    }

    void scan_neighbors(
        float* __restrict__ appro_dist,
        const uint8_t* __restrict__ query_code,
        float query_norm_sqr,
        float query_norm,
        float vl,
        float width,
        int32_t sumq,
        const uint8_t* compact_code,
        const float* factor
    ) const {
        (void)sumq;
        std::vector<float> result_float(degree_bound_);
        for (size_t row = 0; row < degree_bound_; ++row) {
            const uint8_t* code = compact_code + row * (padded_dim_ / 2);
            uint32_t dot = 0;
            for (size_t dim = 0; dim < padded_dim_; dim += 2) {
                const uint8_t packed = code[dim / 2];
                dot += static_cast<uint32_t>(query_code[dim]) *
                       static_cast<uint32_t>(packed & 0x0F);
                dot += static_cast<uint32_t>(query_code[dim + 1]) *
                       static_cast<uint32_t>(packed >> 4);
            }
            result_float[row] = static_cast<float>(dot);
        }
        const float* data_norm_sqr = factor;
        const float* fac_dq = &data_norm_sqr[degree_bound_];
        const float* fac_vq = &fac_dq[degree_bound_];
        appro_dist_impl(
            degree_bound_,
            query_norm_sqr,
            query_norm,
            width,
            vl,
            result_float.data(),
            data_norm_sqr,
            fac_dq,
            fac_vq,
            appro_dist
        );
    }
};
}  // namespace symqg
