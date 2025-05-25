#include "../include/TradingNetwork.h"

TradingNetwork::TradingNetwork(int input_dim, 
                             const std::vector<int>& hidden_dims,
                             bool multi_preference)
    : multi_preference(multi_preference) {
    
    // If multi-preference, add 2 more inputs for A and phi
    if (multi_preference) {
        input_dim += 2;
    }
    
    // Create sequential model
    torch::nn::Sequential layers;
    int prev_dim = input_dim;
    
    for (int dim : hidden_dims) {
        layers->push_back(torch::nn::Linear(prev_dim, dim));
        layers->push_back(torch::nn::Tanh());
        layers->push_back(torch::nn::Dropout(0.2));
        prev_dim = dim;
    }
    
    // Output layer
    layers->push_back(torch::nn::Linear(prev_dim, 1));
    
    // Register the model
    model = layers;
    register_module("model", model);
}

torch::Tensor TradingNetwork::forward(torch::Tensor x) {
    return model->forward(x);
}
