#include <iostream>
#include "PrimitivePipeline.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "ModelBuilder.hpp"

using namespace dnnl;

int main() {
    // Initialize DNNL engine and stream
    engine eng(engine::kind::cpu, 0);
    stream strm(eng);

    // Build and execute model pipeline
    // printf("Memory initialized\n");
    PrimitivePipeline model = build_model_pipeline(eng);
    model.execute(eng, strm);

    std::cout << "Model execution completed successfully." << std::endl;
    return 0;
}
