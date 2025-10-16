#pragma once
// Umbrella header to simplify includes from bindings and examples.

// Core
#include "ag/core/variables.hpp"
#include "ag/ops/graph.hpp"

// Ops
#include "ag/ops/activations.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/linalg.hpp"
#include "ag/ops/reduce.hpp"
#include "ag/ops/reshape.hpp"

// NN
#include "ag/nn/module.hpp"
#include "ag/nn/sequential.hpp"
#include "ag/nn/layers/conv2d.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/layers/normalization.hpp"

// Optional NN extras (unconditional includes)
#include "ag/nn/layers/pooling.hpp"
#include "ag/nn/layers/dropout.hpp"
#include "ag/nn/layers/lstm.hpp"

#include "ag/nn/loss.hpp"
#include "ag/nn/optim/sgd.hpp"

// Data / utils (optional)
#include "ag/data/dataloader.hpp"

// End of umbrella
