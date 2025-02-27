#ifndef MODEL_BUILDER_HPP
#define MODEL_BUILDER_HPP

#include "PrimitivePipeline.hpp"
#include "tensor_utils.h"

// Function to build the model pipeline
PrimitivePipeline build_model_pipeline(dnnl::engine& eng);

#endif // MODEL_BUILDER_HPP
