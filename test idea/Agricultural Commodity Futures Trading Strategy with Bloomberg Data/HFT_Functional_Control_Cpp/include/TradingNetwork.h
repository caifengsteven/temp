#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>

/**
 * @class TradingNetwork
 * @brief Neural network for trading controller
 */
class TradingNetwork : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * @param input_dim Dimension of the input (2 for t and q)
     * @param hidden_dims Dimensions of hidden layers
     * @param multi_preference Whether to include risk aversion parameters as inputs
     */
    TradingNetwork(int input_dim = 2, 
                  const std::vector<int>& hidden_dims = {5, 5, 5},
                  bool multi_preference = false);

    /**
     * @brief Forward pass
     * @param x Input tensor
     * @return Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Check if the network is multi-preference
     * @return True if multi-preference, false otherwise
     */
    bool isMultiPreference() const { return multi_preference; }

private:
    bool multi_preference;
    torch::nn::Sequential model{nullptr};
};
