#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
class QGQuery {
   private:
    const float* query_data_ = nullptr;
    const float* center_ = nullptr;
    std::vector<float, memory::AlignedAllocator<float>> normalized_residual_;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
    size_t dimension_ = 0;
    size_t padded_dim_ = 0;
    float width_ = 0;
    float lower_val_ = 0;
    float upper_val_ = 0;
    float query_norm_ = 0;
    float query_norm_sqr_ = 0;
    int32_t sumq_ = 0;

   public:
    explicit QGQuery(
        const float* q, const float* center, size_t dimension, size_t padded_dim
    )
        : query_data_(q)
        , center_(center)
        , normalized_residual_(dimension)
        , lut_(padded_dim << 2)  // padded_dim / 4 * 16
        , dimension_(dimension)
        , padded_dim_(padded_dim) {}

    void query_prepare(const FHTRotator& rotator, const QGScanner& scanner) {
        query_norm_sqr_ = 0.F;
        for (size_t i = 0; i < dimension_; ++i) {
            float residual = query_data_[i] - center_[i];
            normalized_residual_[i] = residual;
            query_norm_sqr_ += residual * residual;
        }
        query_norm_ = std::sqrt(query_norm_sqr_);
        if (query_norm_ > 0.F) {
            float inv_norm = 1.F / query_norm_;
            for (size_t i = 0; i < dimension_; ++i) {
                normalized_residual_[i] *= inv_norm;
            }
        }

        // rotate query
        std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
        rotator.rotate(normalized_residual_.data(), rd_query.data());

        // quantize query
        std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
        scalar::data_range(rd_query.data(), padded_dim_, lower_val_, upper_val_);
        width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
        if (width_ <= 0.F) {
            lower_val_ = 0.F;
            width_ = 1.F;
        }
        scalar::quantize(
            byte_query.data(), rd_query.data(), padded_dim_, lower_val_, width_, sumq_
        );

        // pack lut
        scanner.pack_lut(byte_query.data(), lut_.data());
    }

    [[nodiscard]] const float& width() const { return width_; }

    [[nodiscard]] const float& lower_val() const { return lower_val_; }

    [[nodiscard]] const float& query_norm() const { return query_norm_; }

    [[nodiscard]] const float& query_norm_sqr() const { return query_norm_sqr_; }

    [[nodiscard]] const int32_t& sumq() const { return sumq_; }

    [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>>& lut(
    ) const {
        return lut_;
    }

    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg
