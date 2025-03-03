#include "tensor_utils.h"
#include <random>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

// Create memory descriptor
memory::desc create_memory_desc(const memory::dims& dims, memory::format_tag format) {
    return memory::desc(dims, memory::data_type::f32, format);
}

// Initialize memory and fill with data
memory initialize_memory(const memory::desc& md, engine& eng, std::vector<float>& data) {
    // printf("Memory initialized\n"); 
    memory mem(md, eng);
    // printf("Memory initialized\n"); 
    write_to_dnnl_memory(data.data(), mem);
    return mem;
}

// Fill tensor with random values
void fill_random_data(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (auto& val : data) {
        val = dist(gen);
    }
}
