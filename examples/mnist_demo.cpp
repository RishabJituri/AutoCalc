// ============================
// File: examples/mnist_demo.cpp
// ============================
// Minimal MNIST demo (MLP) for the AutoCalc stack.
// - Loads ubyte IDX files (unzipped) from --data_dir
// - Trains a 784->128->10 MLP with tanh + MSE on one-hot labels
// - Reports train loss and test accuracy
//
// Example usage:
//   ./build/mnist_demo --data_dir data/mnist --epochs 3 --batch 64 --lr 0.1 --train_limit 10000
//
// Expected files in --data_dir (uncompressed):
//   train-images-idx3-ubyte
//   train-labels-idx1-ubyte
//   t10k-images-idx3-ubyte
//   t10k-labels-idx1-ubyte

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#include "ag/core/variables.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/optim/sgd.hpp"
#include "ag/ops/elementwise.hpp"
#include "ag/ops/reduce.hpp"
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"

using ag::Variable;
using ag::nn::Linear;
using ag::optim::SGD;
using ag::reduce_mean;
using ag::sub;
using ag::mul;

namespace {

uint32_t read_u32_be(std::ifstream& f) {
  unsigned char b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (uint32_t(b[0])<<24) | (uint32_t(b[1])<<16) | (uint32_t(b[2])<<8) | uint32_t(b[3]);
}

struct MnistArrays {
  std::size_t N=0, rows=0, cols=0;
  std::vector<uint8_t> images; // size = N*rows*cols
  std::vector<uint8_t> labels; // size = N
};

MnistArrays load_mnist(const std::string& dir, bool train) {
  const std::string img = dir + (train ? "/train-images-idx3-ubyte" : "/t10k-images-idx3-ubyte");
  const std::string lab = dir + (train ? "/train-labels-idx1-ubyte" : "/t10k-labels-idx1-ubyte");

  std::ifstream fi(img, std::ios::binary);
  std::ifstream fl(lab, std::ios::binary);
  if (!fi) throw std::runtime_error("Failed to open " + img);
  if (!fl) throw std::runtime_error("Failed to open " + lab);

  uint32_t magic_i = read_u32_be(fi);
  if (magic_i != 2051) throw std::runtime_error("Bad magic in images file");
  uint32_t magic_l = read_u32_be(fl);
  if (magic_l != 2049) throw std::runtime_error("Bad magic in labels file");

  uint32_t N = read_u32_be(fi);
  uint32_t rows = read_u32_be(fi);
  uint32_t cols = read_u32_be(fi);

  uint32_t Nl = read_u32_be(fl);
  if (Nl != N) throw std::runtime_error("Labels/images count mismatch");

  MnistArrays m;
  m.N = N; m.rows = rows; m.cols = cols;
  m.images.resize(std::size_t(N)*rows*cols);
  m.labels.resize(N);

  fi.read(reinterpret_cast<char*>(m.images.data()), m.images.size());
  fl.read(reinterpret_cast<char*>(m.labels.data()), m.labels.size());
  if (!fi || !fl) throw std::runtime_error("Truncated MNIST file(s)");
  return m;
}

// One-hot vector for label in [0,9]
std::vector<double> one_hot(int label, std::size_t C=10) {
  std::vector<double> v(C, 0.0);
  if (label>=0 && (std::size_t)label<C) v[(std::size_t)label] = 1.0;
  return v;
}

// Tanh using existing elementwise ops
Variable tanh_v(const Variable& x) {
  auto two = Variable(std::vector<double>(x.value().size(), 2.0), x.shape(), /*requires_grad=*/false);
  auto e2x = expv(mul(two, x));
  auto one = Variable(std::vector<double>(x.value().size(), 1.0), x.shape(), /*requires_grad=*/false);
  return div(sub(e2x, one), add(e2x, one));
}

// A tiny MLP for MNIST (flattened 28x28 -> 128 -> 10)
struct MnistMLP : ag::nn::Module {
  Linear l1, l2;

  MnistMLP()
  : l1(784, 128, /*bias=*/true, 0.02, 1234ull),
    l2(128, 10,  /*bias=*/true, 0.02, 5678ull) {
    register_module("l1", l1);
    register_module("l2", l2);
  }

  Variable forward(const Variable& x) override {
    auto h = l1.forward(x);
    // tanh_v from the demo file
    h = tanh_v(h);
    return l2.forward(h);
  }

protected:
  // Needed because Module::_parameters() is pure virtual
  std::vector<ag::Variable*> _parameters() override { return {}; }

  // Optional: satisfy hooks (no-op)
  void on_mode_change() override {}
};


// Dataset that lazily converts uint8 image to double [0,1] and one-hot label
struct MnistDataset : ag::data::Dataset {
  MnistArrays m;
  bool flatten = true; // always true here
  explicit MnistDataset(MnistArrays arr) : m(std::move(arr)) {}
  std::size_t size() const override { return m.N; }
  ag::data::Example get(std::size_t idx) const override {
    const std::size_t HW = m.rows*m.cols;
    const uint8_t* px = m.images.data() + idx*HW;
    std::vector<double> x(HW);
    for (std::size_t i=0;i<HW;++i) x[i] = double(px[i]) / 255.0;
    auto X = Variable(x, {1,HW}, /*requires_grad=*/false); // per-sample [1,784]
    auto Y = Variable(one_hot((int)m.labels[idx]), {1,10}, /*requires_grad=*/false);
    return {X, Y};
  }
};

struct Args {
  std::string data_dir = "data/mnist";
  int epochs = 3;
  int batch = 64;
  double lr = 0.1;
  int train_limit = -1; // -1 = all
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i) {
    std::string s = argv[i];
    auto need = [&](int i){ if (i+1>=argc) throw std::runtime_error("Missing value for " + s); };
    if (s=="--data_dir") { need(i); a.data_dir = argv[++i]; }
    else if (s=="--epochs") { need(i); a.epochs = std::atoi(argv[++i]); }
    else if (s=="--batch") { need(i); a.batch = std::atoi(argv[++i]); }
    else if (s=="--lr") { need(i); a.lr = std::atof(argv[++i]); }
    else if (s=="--train_limit") { need(i); a.train_limit = std::atoi(argv[++i]); }
    else { std::fprintf(stderr, "Unknown arg: %s\n", s.c_str()); std::exit(2); }
  }
  return a;
}

int argmax10(const double* p) {
  int arg=0; double best=p[0];
  for (int i=1;i<10;++i) if (p[i]>best) { best=p[i]; arg=i; }
  return arg;
}

} // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::printf("Loading MNIST from %s ...\n", args.data_dir.c_str());
    auto train = load_mnist(args.data_dir, /*train=*/true);
    auto test  = load_mnist(args.data_dir, /*train=*/false);

    if (args.train_limit > 0 && (std::size_t)args.train_limit < train.N) {
      train.N = (std::size_t)args.train_limit;
      train.images.resize(train.N * train.rows * train.cols);
      train.labels.resize(train.N);
    }

    MnistDataset dstrain(std::move(train));
    MnistDataset dstest (std::move(test));

    ag::data::DataLoaderOptions opt;
    opt.batch_size = (std::size_t)std::max(1, args.batch);
    opt.shuffle = true;
    opt.drop_last = true;
    opt.seed = 12345ull;

    ag::data::DataLoader loader(dstrain, opt);

    MnistMLP model;
    SGD optm(args.lr);

    for (int epoch=1; epoch<=args.epochs; ++epoch) {
      loader.reset();
      double running = 0.0;
      std::size_t batches = 0;
      auto t0 = std::chrono::steady_clock::now();
      while (loader.has_next()) {
        auto batch = loader.next(); // x:[B,1,784]? we built as [1,784] per sample, collate -> [B,1,784]
        // Flatten away the singleton dimension to [B,784]:
        // We can reconstruct a flat Variable by copying values and setting shape.
        const auto& xv = batch.x.value();
        const std::size_t B = batch.size;
        const std::size_t HW = 784;
        Variable X(xv, {B, HW}, /*requires_grad=*/false);
        const auto& yv = batch.y.value();
        Variable Y(yv, {B, 10}, /*requires_grad=*/false);

        auto Yhat = model.forward(X);
        auto diff = sub(Yhat, Y);
        auto L = reduce_mean(mul(diff, diff), /*axes=*/{0,1}, /*keepdims=*/false);
        // backward
        L.backward({1.0});
        optm.step(model);
        model.zero_grad();

        double cur = 0.0; for (double v : L.value()) cur += v;
        running += cur; ++batches;
      }
      auto t1 = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t1-t0).count();
      std::printf("[epoch %d] train loss (mean over batches): %.6f  (%.3fs)\n",
                  epoch, running / std::max<std::size_t>(1,batches), secs);

      // Eval on test set (accuracy by argmax)
      ag::data::DataLoaderOptions topt = opt;
      topt.shuffle = false;
      topt.drop_last = false;
      ag::data::DataLoader testloader(dstest, topt);
      std::size_t correct = 0, total = 0;
      while (testloader.has_next()) {
        auto b = testloader.next();
        const std::size_t B = b.size;
        Variable X(b.x.value(), {B, 784}, /*requires_grad=*/false);
        auto logits = model.forward(X);
        const auto& lv = logits.value();
        for (std::size_t i=0;i<B;++i) {
          int pred = argmax10(&lv[i*10]);
          int gold = int(std::llround(b.y.value()[i*10 + pred] > 0.5 ? pred : std::distance(&b.y.value()[i*10], std::max_element(&b.y.value()[i*10], &b.y.value()[i*10+10]))));
          // Since y is one-hot, the index of max is the label.
          gold = int(std::max_element(&b.y.value()[i*10], &b.y.value()[i*10+10]) - &b.y.value()[i*10]);
          if (pred == gold) ++correct;
          ++total;
        }
      }
      double acc = (total==0) ? 0.0 : double(correct) / double(total);
      std::printf("           test accuracy: %.2f%%  (%zu/%zu)\n", 100.0*acc, correct, total);
    }

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
