#include "ModelBuilder.hpp"  // Ensure this file exists
#include <map>
#include <vector>
#include <iostream>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;


// Helper function to initialize memory objects
std::map<std::string, memory> initialize_memory_objects(
    engine& eng, 
    const std::map<std::string, memory::dims>& tensor_shapes, 
    std::map<std::string, std::vector<float>>& tensor_data) {
    
    std::map<std::string, memory> memory_objects;

    for (const auto& [name, dims] : tensor_shapes) {
        auto mem_desc = create_memory_desc(dims, memory::format_tag::any);
        printf("Memory initialized\n"); 
        memory_objects[name] = initialize_memory(mem_desc, eng, tensor_data[name]);
    }

    return memory_objects;
}

// Helper function to allocate and fill tensor data
std::map<std::string, std::vector<float>> allocate_and_initialize_tensors(
    const std::map<std::string, memory::dims>& tensor_shapes) {
    
    std::map<std::string, std::vector<float>> tensor_data;
    
    for (const auto& [name, dims] : tensor_shapes) {
        tensor_data[name] = std::vector<float>(product(dims));
        fill_random_data(tensor_data[name]);
    }

    return tensor_data;
}

// Helper function to define tensor dimensions
std::map<std::string, memory::dims> define_tensor_shapes() {
    return {
        {"src", {1, 12, 768}},
        {"query", {1, 12, 768}},
        {"key", {1, 12, 768}},
        {"value", {1, 12, 768}},
        {"weight_q", {768, 768}},
        {"weight_k", {768, 768}},
        {"weight_v", {768, 768}},
        {"weight_o", {768, 768}},
        {"ffn_weight1", {768, 3072}},
        {"ffn_weight2", {3072, 768}},
        {"ffn_bias1", {1, 1, 3072}},
        {"ffn_bias2", {1, 1, 768}},
        {"attn_out", {1, 12, 768}},
        {"ffn_out", {1, 12, 768}}
    };
}

// Self-Attention Layer
void build_attention_layer(engine& eng, std::map<std::string, memory>& memory_objects, PrimitivePipeline& model) {
    PrimitivePipeline attention_pipeline;
    
    // Query, Key, Value MatMul
    std::vector<std::string> qkv = {"query", "key", "value"};
    for (const auto& name : qkv) {
        auto matmul_pd = matmul::primitive_desc(eng, 
            memory_objects.at("src").get_desc(),
            memory_objects.at("weight_" + name.substr(0, 1)).get_desc(),
            memory_objects.at(name).get_desc()
        );
        model.insert({matmul(matmul_pd), {
            {DNNL_ARG_SRC, memory_objects.at("src")},
            {DNNL_ARG_WEIGHTS, memory_objects.at("weight_" + name.substr(0, 1))},
            {DNNL_ARG_DST, memory_objects.at(name)}
        }});
    }
    
    // Scaled Dot-Product Attention (softmax on Q*K^T, multiply by V)
    // Implement softmax separately in a real case
    auto attn_matmul_pd = matmul::primitive_desc(eng,
        memory_objects.at("query").get_desc(),
        memory_objects.at("key").get_desc(),
        memory_objects.at("attn_out").get_desc()
    );
    model.insert({matmul(attn_matmul_pd), {
        {DNNL_ARG_SRC, memory_objects.at("query")},
        {DNNL_ARG_WEIGHTS, memory_objects.at("key")},
        {DNNL_ARG_DST, memory_objects.at("attn_out")}
    }});
    
    // return attention_pipeline;
}

// Feedforward Network (FFN)
void build_ffn_layer(engine& eng, std::map<std::string, memory>& memory_objects, PrimitivePipeline& model) {
    PrimitivePipeline ffn_pipeline;
    
    // First MatMul + ReLU
    primitive_attr matmul_attr;
    post_ops matmul_post_ops;
    matmul_post_ops.append_eltwise(algorithm::eltwise_relu, 1.0f, 0.0f);
    matmul_attr.set_post_ops(matmul_post_ops);
    
    auto ffn1_pd = matmul::primitive_desc(eng, 
        memory_objects.at("attn_out").get_desc(),
        memory_objects.at("ffn_weight1").get_desc(),
        memory_objects.at("ffn_out").get_desc(),
        matmul_attr
    );
    model.insert({matmul(ffn1_pd), {
        {DNNL_ARG_SRC, memory_objects.at("attn_out")},
        {DNNL_ARG_WEIGHTS, memory_objects.at("ffn_weight1")},
        {DNNL_ARG_BIAS, memory_objects.at("ffn_bias1")},
        {DNNL_ARG_DST, memory_objects.at("ffn_out")}
    }});
    
    // Second MatMul
    auto ffn2_pd = matmul::primitive_desc(eng, 
        memory_objects.at("ffn_out").get_desc(),
        memory_objects.at("ffn_weight2").get_desc(),
        memory_objects.at("src").get_desc()
    );
    model.insert({matmul(ffn2_pd), {
        {DNNL_ARG_SRC, memory_objects.at("ffn_out")},
        {DNNL_ARG_WEIGHTS, memory_objects.at("ffn_weight2")},
        {DNNL_ARG_BIAS, memory_objects.at("ffn_bias2")},
        {DNNL_ARG_DST, memory_objects.at("src")}
    }});
    
    // return ffn_pipeline;
}

// PrimitivePipeline build_moe_layer(engine& eng, std::map<std::string, memory>& memory_objects, int num_experts) {
//     PrimitivePipeline moe_pipeline; 
    
//     // Gating mechanism (MatMul)
//     primitive_attr gate_attr;
//     post_ops gate_post_ops;
//     gate_attr.set_post_ops(gate_post_ops);
    
//     auto gate_pd = matmul::primitive_desc(eng, 
//         memory_objects.at("attn_out").get_desc(),
//         memory_objects.at("gate_weight").get_desc(),
//         memory_objects.at("gate_out").get_desc()
//     );
    
//     moe_pipeline.push_back({matmul(gate_pd), {
//         {DNNL_ARG_SRC, memory_objects.at("attn_out")},
//         {DNNL_ARG_WEIGHTS, memory_objects.at("gate_weight")},
//         {DNNL_ARG_BIAS, memory_objects.at("gate_bias")},
//         {DNNL_ARG_DST, memory_objects.at("gate_out")}
//     }});

//     // Experts computation (parallel MatMul + ReLU)
//     std::vector<memory> expert_outputs(num_experts);
//     std::vector<memory::desc> expert_mds;
//     std::vector<float> scales(num_experts, 1.0f);
    
//     for (int i = 0; i < num_experts; i++) {
//         primitive_attr expert_attr;
//         post_ops expert_post_ops;
//         expert_post_ops.append_eltwise(algorithm::eltwise_relu, 1.0f, 0.0f);
//         expert_attr.set_post_ops(expert_post_ops);
        
//         auto expert_pd = matmul::primitive_desc(eng, 
//             memory_objects.at("attn_out").get_desc(),
//             memory_objects.at("expert_weight" + std::to_string(i)).get_desc(),
//             memory_objects.at("expert_out" + std::to_string(i)).get_desc(),
//             expert_attr
//         );

//         moe_pipeline.push_back({matmul(expert_pd), {
//             {DNNL_ARG_SRC, memory_objects.at("attn_out")},
//             {DNNL_ARG_WEIGHTS, memory_objects.at("expert_weight" + std::to_string(i))},
//             {DNNL_ARG_BIAS, memory_objects.at("expert_bias" + std::to_string(i))},
//             {DNNL_ARG_DST, memory_objects.at("expert_out" + std::to_string(i))}
//         }});

//         expert_outputs[i] = memory_objects.at("expert_out" + std::to_string(i));
//         expert_mds.push_back(expert_outputs[i].get_desc());
//     }

//     // Define output memory descriptor
//     auto moe_out_md = memory_objects.at("moe_out").get_desc();

//     // Create sum primitive descriptor with correct argument order
//     auto weighted_sum_pd = sum::primitive_desc(eng, scales, expert_mds, moe_out_md);
//     auto weighted_sum_prim = sum(weighted_sum_pd);

//     // Add sum operation to pipeline
//     std::unordered_map<int, memory> sum_inputs;
//     for (int i = 0; i < num_experts; i++) {
//         sum_inputs[DNNL_ARG_MULTIPLE_SRC + i] = expert_outputs[i];
//     }
//     sum_inputs[DNNL_ARG_DST] = memory_objects.at("moe_out");

//     moe_pipeline.push_back({weighted_sum_prim, sum_inputs});

//     return moe_pipeline;
// }

// Main function to build the model pipeline
PrimitivePipeline build_model_pipeline(engine& eng) {
    stream strm(eng);
    
    
    auto tensor_shapes = define_tensor_shapes();
    auto tensor_data = allocate_and_initialize_tensors(tensor_shapes);
    // printf("Memory initialized\n");
    auto memory_objects = initialize_memory_objects(eng, tensor_shapes, tensor_data);
    // printf("Memory initialized\n");
    PrimitivePipeline model;
    build_attention_layer(eng, memory_objects, model);
    build_ffn_layer(eng, memory_objects, model);
    // auto moe_layer = build_moe_layer(eng, memory_objects, 4);
    
    // model.insert(attention_layer);
    // model.insert(ffn_layer);
    // model.insert(model.end(), moe_layer.begin(), moe_layer.end());
    return model;
}

