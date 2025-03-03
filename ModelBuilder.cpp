#include "ModelBuilder.hpp"  // Ensure this file exists
#include <map>
#include <vector>
#include <iostream>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <iostream>
#include <queue>
#include <utility>
#include <dnnl.hpp>


using namespace dnnl;

memory::format_tag get_format_tag(const std::vector<long>& dims) {
    switch (dims.size()) {
        case 1: return memory::format_tag::a;
        case 2: return memory::format_tag::ab;
        case 3: return memory::format_tag::abc;
        case 4: return memory::format_tag::abcd;
        case 5: return memory::format_tag::abcde;
        case 6: return memory::format_tag::abcdef;
        default: return memory::format_tag::any; // Default case
    }
}

// Helper function to initialize memory objects
std::map<std::string, memory> initialize_memory_objects(
    engine& eng, 
    const std::map<std::string, memory::dims>& tensor_shapes, 
    std::map<std::string, std::vector<float>>& tensor_data) {
    
    std::map<std::string, memory> memory_objects;

    for (const auto& [name, dims] : tensor_shapes) {
        std::cout << "Initializing memory for " << name << " tensor" << " with dims: ";
        for (auto dim : dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        auto format_tag = get_format_tag(dims);
        auto mem_desc = create_memory_desc(dims, format_tag);
        // printf("Memory initialized\n"); 
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
        {"weight_q", {1, 768, 768}},
        {"weight_k", {1, 768, 768}},
        {"weight_v", {1, 768, 768}},
        {"weight_o", {1, 768, 768}},
        {"ffn_weight1", {1, 768, 3072}},
        {"ffn_weight2", {1, 3072, 768}},
        {"ffn_bias1", {1, 1, 3072}},
        {"ffn_bias2", {1, 1, 768}},
        {"attn_out", {1, 12, 768}},
        {"ffn_out", {1, 12, 3072}},
        {"gate_weight", {1, 768, 4}},
        {"gate_out", {1, 12, 4}},
        {"expert_weight0", {1, 768, 768}},
        {"expert_bias0", {1, 1, 768}},
        {"expert_out0", {1, 12, 768}},
        {"expert_weight1", {1, 768, 768}},
        {"expert_bias1", {1, 1, 768}},
        {"expert_out1", {1, 12, 768}},
        {"expert_weight2", {1, 768, 768}},
        {"expert_bias2", {1, 1, 768}},
        {"expert_out2", {1, 12, 768}},
        {"expert_weight3", {1, 768, 768}},
        {"expert_bias3", {1, 1, 768}},
        {"expert_out3", {1, 12, 768}},
        {"moe_out", {1, 12, 768}}
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
    printf("Softmax executed\n");
    // Scaled Dot-Product Attention (softmax on Q*K^T, multiply by V)
    // Implement softmax separately in a real case
    // auto attn_matmul_pd = matmul::primitive_desc(eng,
    //     memory_objects.at("query").get_desc(),
    //     memory_objects.at("key").get_desc(),
    //     memory_objects.at("attn_out").get_desc()
    // );
    // model.insert({matmul(attn_matmul_pd), {
    //     {DNNL_ARG_SRC, memory_objects.at("query")},
    //     {DNNL_ARG_WEIGHTS, memory_objects.at("key")},
    //     {DNNL_ARG_DST, memory_objects.at("attn_out")}
    // }});

    auto softmax_pd = softmax_forward::primitive_desc(eng,
        prop_kind::forward_inference, algorithm::softmax_accurate, memory_objects.at("attn_out").get_desc(),
        memory_objects.at("attn_out").get_desc(), /* axis = */ memory_objects.at("attn_out").get_desc().get_ndims() - 1 );
    
    auto attn_softmax_pd = softmax_forward(softmax_pd);
        
    model.insert({attn_softmax_pd, {
        {DNNL_ARG_SRC, memory_objects.at("attn_out")},
        {DNNL_ARG_DST, memory_objects.at("attn_out")}
    }});

    
    // auto attn_matmul2_pd = matmul::primitive_desc(eng,
    //     memory_objects.at("attn_out").get_desc(),
    //     memory_objects.at("value").get_desc(),
    //     memory_objects.at("src").get_desc()
    // );
    // model.insert({matmul(attn_matmul2_pd), {
    //     {DNNL_ARG_SRC, memory_objects.at("attn_out")},
    //     {DNNL_ARG_WEIGHTS, memory_objects.at("value")},
    //     {DNNL_ARG_DST, memory_objects.at("src")}
    // }});
    
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
        memory_objects.at("src").get_desc(),
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

std::vector<std::vector<int>> select_top_k_experts_per_token(
    const std::vector<float>& gating_scores, int num_tokens, int num_experts, int k) {

    std::vector<std::vector<int>> top_k_experts_per_token(num_tokens);

    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        std::priority_queue<std::pair<float, int>> max_heap;

        int offset = token_idx * num_experts;  
        for (int i = 0; i < num_experts; i++) {
            max_heap.push({gating_scores[offset + i], i});
        }

        for (int i = 0; i < k; i++) {
            top_k_experts_per_token[token_idx].push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    return top_k_experts_per_token;
}


void build_moe_layer(engine& eng, std::map<std::string, memory>& memory_objects, 
    int num_experts, int k, PrimitivePipeline& model) {

    printf("[DEBUG] Starting MoE Layer Construction\n");

    if (k > num_experts) {
        printf("[ERROR] k (%d) cannot be greater than num_experts (%d)\n", k, num_experts);
        throw std::invalid_argument("k cannot be greater than num_experts");
    }

    dnnl::stream s(eng);

    // Gating mechanism (MatMul)
    auto gate_pd = matmul::primitive_desc(eng, 
        memory_objects.at("src").get_desc(),
        memory_objects.at("gate_weight").get_desc(),
        memory_objects.at("gate_out").get_desc()
    );

    model.insert({matmul(gate_pd), {
        {DNNL_ARG_SRC, memory_objects.at("src")},
        {DNNL_ARG_WEIGHTS, memory_objects.at("gate_weight")},
        {DNNL_ARG_DST, memory_objects.at("gate_out")}
    }});

    printf("[DEBUG] Gating executed\n");

    // Ensure `selected_experts` has the correct size
    std::vector<std::vector<int>> selected_experts(12, std::vector<int>(k, 0));

    // Insert custom function: Select top-K experts
    model.insert_custom([&]() {
        printf("[DEBUG] Selecting top-%d experts per token\n", k);

        // Read gating scores from memory
        std::vector<float> gating_scores(12 * num_experts, 0.0f);
        read_from_dnnl_memory(gating_scores.data(), memory_objects.at("gate_out"));

        selected_experts = select_top_k_experts_per_token(gating_scores, 12, num_experts, k);

        if (selected_experts.size() != 12 || selected_experts[0].size() != k) {
            printf("[ERROR] Invalid selected_experts size: %zu x %zu\n", 
                   selected_experts.size(), 
                   selected_experts.empty() ? 0 : selected_experts[0].size());
            throw std::runtime_error("Invalid selected_experts size");
        }

        printf("[DEBUG] Selected Experts per token:\n");
        for (int token = 0; token < 12; token++) {
            printf("Token %d: ", token);
            for (int expert : selected_experts[token]) {
                printf("%d ", expert);
            }
            printf("\n");
        }
    });

    printf("[DEBUG] Inserted custom function into pipeline\n");

    // Experts computation (MatMul + ReLU only for top-K per token)
    model.insert_custom([&]() {
        printf("[DEBUG] Executing expert computations\n");

        for (int token_idx = 0; token_idx < 12; token_idx++) {
            for (int i = 0; i < k; i++) {
                int expert_idx = selected_experts[token_idx][i];

                primitive_attr expert_attr;
                post_ops expert_post_ops;
                expert_post_ops.append_eltwise(algorithm::eltwise_relu, 1.0f, 0.0f);
                expert_attr.set_post_ops(expert_post_ops);

                std::string expert_key = "expert_out" + std::to_string(expert_idx);
                std::string expert_weight_key = "expert_weight" + std::to_string(expert_idx);

                printf("[DEBUG] Processing Expert %d for Token %d\n", expert_idx, token_idx);

                auto expert_pd = matmul::primitive_desc(eng, 
                    memory_objects.at("src").get_desc(),
                    memory_objects.at(expert_weight_key).get_desc(),
                    memory_objects.at(expert_key).get_desc(),
                    expert_attr
                );

                model.insert({matmul(expert_pd), {
                    {DNNL_ARG_SRC, memory_objects.at("src")},
                    {DNNL_ARG_WEIGHTS, memory_objects.at(expert_weight_key)},
                    {DNNL_ARG_BIAS, memory_objects.at("expert_bias" + std::to_string(expert_idx))},
                    {DNNL_ARG_DST, memory_objects.at(expert_key)}
                }});

                printf("[DEBUG] Inserted MatMul for Expert %d, Token %d\n", expert_idx, token_idx);
            }
        }
    });

    printf("[DEBUG] MoE Layer Built with Top-%d Experts Per Token\n", k);
}

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
    build_moe_layer(eng, memory_objects, 4, 1, model);
    // auto moe_layer = build_moe_layer(eng, memory_objects, 4);
    
    // model.insert(attention_layer);
    // model.insert(ffn_layer);
    // model.insert(model.end(), moe_layer.begin(), moe_layer.end());
    return model;
}

