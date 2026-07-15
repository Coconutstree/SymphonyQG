#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"
#include "utils/io.hpp"

#ifndef SYMQG_DEGREE
#define SYMQG_DEGREE 32
#endif

#ifndef SYMQG_EF_BUILD
#define SYMQG_EF_BUILD 400
#endif

#ifndef SYMQG_NUM_ITER
#define SYMQG_NUM_ITER 3
#endif

#ifndef SYMQG_TOPK
#define SYMQG_TOPK 1
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(const Clock::time_point start, const Clock::time_point end) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    );
}

double file_size_mb(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return 0.0;
    }
    return static_cast<double>(std::filesystem::file_size(path)) / (1024.0 * 1024.0);
}

std::pair<size_t, size_t> vecs_shape(const std::string& path, size_t element_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "File " << path << " not exists\n";
        std::abort();
    }
    uint32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    const size_t file_size = std::filesystem::file_size(path);
    const size_t row_bytes = sizeof(uint32_t) + (static_cast<size_t>(dim) * element_size);
    if (dim == 0 || row_bytes == 0 || file_size % row_bytes != 0) {
        std::cerr << "Invalid vecs file shape: " << path << '\n';
        std::abort();
    }
    return {file_size / row_bytes, dim};
}

std::vector<size_t> ef_sweep() {
    std::vector<size_t> efs;
    for (size_t ef = 1; ef <= 30; ++ef) {
        efs.push_back(ef);
    }
    for (size_t ef = 40; ef <= 100; ef += 10) {
        efs.push_back(ef);
    }
    for (size_t ef = 140; ef <= 460; ef += 40) {
        efs.push_back(ef);
    }
    return efs;
}

}  // namespace

int main() {
    using FloatMatrix = symqg::RowMatrix<float>;
    using IntMatrix = symqg::RowMatrix<int>;

    const std::string dataset = "dbpedia_openai1536";
    const std::string data_dir = "/home/kai3/coco/data/" + dataset;
    const std::string base_path = data_dir + "/" + dataset + "_base.fvecs";
    const std::string query_path = data_dir + "/" + dataset + "_query.fvecs";
    const std::string gt_path = data_dir + "/" + dataset + "_groundtruth.ivecs";
    const size_t degree = SYMQG_DEGREE;
    const size_t ef_build = SYMQG_EF_BUILD;
    const size_t num_iter = SYMQG_NUM_ITER;
    const uint32_t topk = SYMQG_TOPK;
    const std::string index_path = dataset + "_symphonyqg_D" +
                                   std::to_string(degree) + "_efbuild_" +
                                   std::to_string(ef_build) + "_iter_" +
                                   std::to_string(num_iter) + ".index";
    const auto [base_count, base_dim] = vecs_shape(base_path, sizeof(float));

    std::cout << std::setprecision(6);
    std::cout << "Run config:\n";
    std::cout << "  dataset=" << dataset << '\n';
    std::cout << "  base_count=" << base_count << '\n';
    std::cout << "  degree=" << degree << " ef_build=" << ef_build
              << " num_iter=" << num_iter << " topk=" << topk << '\n';
    std::cout << "  graph=symphonyqg metric=l2 stored_data=raw_float_and_qg_codes\n";
    std::cout << "  base_path=" << base_path << '\n';
    std::cout << "  query_path=" << query_path << '\n';
    std::cout << "  gt_path=" << gt_path << '\n';
    std::cout << "  index_path=" << index_path << '\n';

    FloatMatrix query;
    IntMatrix gt;

    auto stage_start = Clock::now();
    symqg::load_vecs<float, FloatMatrix>(query_path.c_str(), query);
    auto stage_end = Clock::now();
    std::cout << "build_stage=load_query us=" << elapsed_us(stage_start, stage_end)
              << " count=" << query.rows() << " dim=" << query.cols() << '\n';

    stage_start = Clock::now();
    symqg::load_vecs<int, IntMatrix>(gt_path.c_str(), gt);
    stage_end = Clock::now();
    std::cout << "build_stage=load_gt us=" << elapsed_us(stage_start, stage_end)
              << " count=" << gt.rows() << " dim=" << gt.cols() << '\n';

    if (query.rows() != gt.rows()) {
        std::cerr << "Query/GT row count mismatch\n";
        return 1;
    }
    if (gt.cols() < topk) {
        std::cerr << "GT top-k is smaller than requested topk\n";
        return 1;
    }
    if (static_cast<size_t>(query.cols()) != base_dim) {
        std::cerr << "Base/query dimension mismatch\n";
        return 1;
    }

    symqg::QuantizedGraph qg(base_count, degree, base_dim);

    if (std::filesystem::exists(index_path)) {
        std::cout << "Loading index from " << index_path << ":\n";
        stage_start = Clock::now();
        qg.load_index(index_path.c_str());
        stage_end = Clock::now();
        std::cout << "build_stage=load_index us=" << elapsed_us(stage_start, stage_end)
                  << '\n';
    } else {
        FloatMatrix base;
        auto total_start = Clock::now();

        std::cout << "Building SymphonyQG index:\n";
        stage_start = Clock::now();
        symqg::load_vecs<float, FloatMatrix>(base_path.c_str(), base);
        stage_end = Clock::now();
        const double load_base_us = elapsed_us(stage_start, stage_end);
        std::cout << "build_stage=load_base us=" << load_base_us
                  << " count=" << base.rows() << " dim=" << base.cols() << '\n';

        if (base.cols() != query.cols()) {
            std::cerr << "Base/query dimension mismatch\n";
            return 1;
        }

        stage_start = Clock::now();
        symqg::QGBuilder builder(qg, ef_build, base.data(), 9999);
        stage_end = Clock::now();
        const double init_copy_us = elapsed_us(stage_start, stage_end);
        std::cout << "build_stage=qg_init_copy us=" << init_copy_us << '\n';

        stage_start = Clock::now();
        builder.build(num_iter);
        stage_end = Clock::now();
        const double graph_build_us = elapsed_us(stage_start, stage_end);
        std::cout << "build_stage=qg_graph_build us=" << graph_build_us
                  << " kips="
                  << (static_cast<double>(base.rows()) * 1000.0 / graph_build_us)
                  << '\n';

        stage_start = Clock::now();
        qg.save_index(index_path.c_str());
        stage_end = Clock::now();
        const double save_index_us = elapsed_us(stage_start, stage_end);
        std::cout << "build_stage=save_index us=" << save_index_us << '\n';

        auto total_end = Clock::now();
        std::cout << "build_total_us=" << elapsed_us(total_start, total_end) << '\n';
    }

    std::cout << "Index storage size: " << file_size_mb(index_path)
              << " MB (index=" << file_size_mb(index_path)
              << " MB, total_bytes="
              << (std::filesystem::exists(index_path) ? std::filesystem::file_size(index_path)
                                                      : 0)
              << ")\n";

    std::vector<uint32_t> results(static_cast<size_t>(query.rows()) * topk);
    for (size_t ef : ef_sweep()) {
        qg.set_ef(ef);
        auto search_start = Clock::now();
        for (long i = 0; i < query.rows(); ++i) {
            qg.search(&query(i, 0), topk, results.data() + (static_cast<size_t>(i) * topk));
        }
        auto search_end = Clock::now();
        const double total_search_us = elapsed_us(search_start, search_end);

        size_t correct = 0;
        for (long i = 0; i < query.rows(); ++i) {
            for (uint32_t r = 0; r < topk; ++r) {
                const uint32_t found = results[static_cast<size_t>(i) * topk + r];
                for (uint32_t g = 0; g < topk; ++g) {
                    if (found == static_cast<uint32_t>(gt(i, g))) {
                        ++correct;
                        break;
                    }
                }
            }
        }

        const double recall = static_cast<double>(correct) /
                              static_cast<double>(query.rows() * topk);
        const double us_per_query = total_search_us / static_cast<double>(query.rows());
        std::cout << ef << '\t' << recall << '\t' << us_per_query
                  << " us\tsymphonyqg_search_us_per_query=" << us_per_query
                  << "\ttotal_us_per_query=" << us_per_query << '\n';
    }

    return 0;
}
