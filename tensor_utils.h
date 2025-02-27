#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <vector>
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

// Function to create memory descriptor
memory::desc create_memory_desc(const memory::dims& dims, memory::format_tag format = memory::format_tag::abc);

// Function to initialize memory
memory initialize_memory(const memory::desc& md, engine& eng, std::vector<float>& data);

// Function to fill a tensor with random values
void fill_random_data(std::vector<float>& data);

#endif // TENSOR_UTILS_H
