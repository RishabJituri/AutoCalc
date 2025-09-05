#pragma once
#include "ag/nn/module.hpp"

namespace ag::nn {

struct MaxPool2d : Module {
  std::size_t kH, kW, sH, sW, pH, pW;
  MaxPool2d(std::size_t kH, std::size_t kW, std::size_t sH=0, std::size_t sW=0,
            std::size_t pH=0, std::size_t pW=0)
    : kH(kH), kW(kW), sH(sH? sH : kH), sW(sW? sW : kW), pH(pH), pW(pW) {}

  Variable forward(const Variable& x) override;
  protected:
    // MaxPool2d has no trainable parameters
    std::vector<ag::Variable*> _parameters() override { return {}; }  
};

struct AvgPool2d : Module {
  std::size_t kH, kW, sH, sW, pH, pW;
  AvgPool2d(std::size_t kH, std::size_t kW, std::size_t sH=0, std::size_t sW=0,
            std::size_t pH=0, std::size_t pW=0)
    : kH(kH), kW(kW), sH(sH? sH : kH), sW(sW? sW : kW), pH(pH), pW(pW) {}

  Variable forward(const Variable& x) override;
  protected:
    // AvgPool2d has no trainable parameters
    std::vector<ag::Variable*> _parameters() override { return {}; }
};

} // namespace ag::nn
