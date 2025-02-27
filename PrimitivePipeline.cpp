#include "PrimitivePipeline.hpp"

void PrimitivePipeline::insert(const MatMulOperation& op) {
            operations.push_back(op);
        }
    
void PrimitivePipeline::execute(dnnl::engine& eng, dnnl::stream& strm) {
            for (auto& op : operations) {
                op.primitive.execute(strm, op.args);
            }
        }
    
 