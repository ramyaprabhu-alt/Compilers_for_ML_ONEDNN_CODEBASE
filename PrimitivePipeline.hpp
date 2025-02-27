#ifndef PRIMITIVE_PIPELINE_HPP
#define PRIMITIVE_PIPELINE_HPP

#include "oneapi/dnnl/dnnl.hpp"
#include <vector>
#include <unordered_map>


// Structure for a matrix multiplication operation (Keep this here)
struct MatMulOperation {
    dnnl::matmul primitive;
    std::unordered_map<int, dnnl::memory> args;
};

// Class to manage a sequence of operations
class PrimitivePipeline {
public:
    void insert(const MatMulOperation& op);
    void execute(dnnl::engine& eng, dnnl::stream& strm);
    void append(const PrimitivePipeline& other);
    MatMulOperation* get_last_operation() {
        return operations.empty() ? nullptr : &operations.back();
    }
private:
    std::vector<MatMulOperation> operations;
};

#endif  // PRIMITIVE_PIPELINE_HPP
