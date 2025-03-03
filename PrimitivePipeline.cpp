#include "PrimitivePipeline.hpp"

void PrimitivePipeline::insert(const MatMulOperation& op) {
            operations.push_back(op);
        }
    
void PrimitivePipeline::execute(dnnl::engine& eng, dnnl::stream& strm) {
    for (auto& op : operations) {
        if (std::holds_alternative<dnnl::primitive>(op.primitive)) {
            std::get<dnnl::primitive>(op.primitive).execute(strm, op.args);
        } else if (std::holds_alternative<std::function<void()>>(op.primitive)) {
            std::get<std::function<void()>>(op.primitive)();
        }
    }
}
    
void PrimitivePipeline::insert_custom(const std::function<void()>& custom_func) {
            operations.push_back({custom_func, {}});
        }
        
        // void PrimitivePipeline::execute(dnnl::engine& eng, dnnl::stream& strm) {
        //     for (auto& op : operations) {
        //         if (std::holds_alternative<dnnl::primitive>(op.operation)) {
        //             std::get<dnnl::primitive>(op.operation).execute(strm, op.args);
        //         } else if (std::holds_alternative<std::function<void()>>(op.operation)) {
        //             std::get<std::function<void()>>(op.operation)();
        //         }
        //     }
        // }
        
        
        