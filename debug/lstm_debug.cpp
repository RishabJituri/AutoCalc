// filepath: debug/lstm_debug.cpp
#include "ag/nn/layers/lstm.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/core/variables.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using ag::Variable;
using ag::nn::LSTMCell;
using ag::nn::LSTM;

static std::vector<float> zeros(std::size_t n){return std::vector<float>(n,0.0f);} 
static std::vector<float> ones(std::size_t n){return std::vector<float>(n,1.0f);} 

float finite_diff_sum_wrt_x(LSTMCell& cell, const std::vector<float>& x0, const Variable& h0, const Variable& c0, std::size_t idx, float eps){
  const std::size_t I = x0.size();
  auto xp = x0; xp[idx] += eps;
  auto xm = x0; xm[idx] -= eps;
  Variable Xp(xp, {1,I}, /*requires_grad=*/false);
  Variable Xm(xm, {1,I}, /*requires_grad=*/false);
  auto hp = cell.forward_step(Xp, h0, c0).first;
  auto hm = cell.forward_step(Xm, h0, c0).first;
  float sp=0.0f, sm=0.0f; for(float v: hp.value()) sp+=v; for(float v: hm.value()) sm+=v;
  return (sp - sm)/(2*eps);
}

int main(){
  const std::size_t I=2, H=2, B=1; 
  LSTMCell cell(I, H, /*bias=*/true, /*init_scale=*/0.05f, /*seed=*/1234ull);

  std::vector<float> xv = {0.1f, -0.2f};
  Variable x(xv, {B,I}, /*requires_grad=*/true);
  Variable h0(zeros(B*H), {B,H}, /*requires_grad=*/true);
  Variable c0(zeros(B*H), {B,H}, /*requires_grad=*/true);

  auto [h1, c1] = cell.forward_step(x,h0,c0);
  // L = sum(h1)
  h1.backward(ones(h1.value().size()));

  // Compare first two grads numerically
  const float eps = 1e-4f;
  for(std::size_t i=0;i<xv.size();++i){
    float gnum = finite_diff_sum_wrt_x(cell, xv, h0, c0, i, eps);
    float gaut = x.grad()[i];
    std::cout << "i="<<i<<" g_autograd="<<gaut<<" g_numeric="<<gnum
              << " diff="<< std::fabs(gaut-gnum) << std::endl;
  }

  // Also test a short unroll of LSTM wrapper on random data
  {
    const std::size_t T=3; 
    std::vector<float> X(B*T*I, 0.0f);
    for(std::size_t t=0;t<T;++t){
      for(std::size_t i=0;i<I;++i) X[(0*T + t)*I + i] = float(t+1)*0.1f*(i? -1.f:1.f);
    }
    Variable Xv(X,{B,T,I}, /*requires_grad=*/false);
    LSTM lstm(I,H,/*num_layers=*/1,/*bias=*/true);
    auto Y = lstm.forward(Xv);
    std::cout << "Y shape: ["<<Y.shape()[0]<<","<<Y.shape()[1]<<","<<Y.shape()[2]<<"]\n";
    std::vector<float> seed(Y.value().size(), 1.0f);
    Y.backward(seed);
    std::cout << "Backward through LSTM wrapper completed." << std::endl;
  }

  return 0;
}
