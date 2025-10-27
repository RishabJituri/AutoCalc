// examples/mnist_demo.cpp — Dataset/DataLoader version
// - Uses ag::data::Dataset + DataLoader
// - Prints per-batch train loss and per-batch test loss
// - Appends summary to results_mnist.txt (timestamp, wall time, test accuracy)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "ag/core/variables.hpp"
#include "ag/ops/reshape.hpp"
#include "ag/ops/activations.hpp"
#include "ag/ops/graph.hpp"
#include "ag/nn/layers/conv2d.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/layers/pooling.hpp"
#include "ag/nn/loss.hpp"
#include "ag/nn/optim/sgd.hpp"   // header path; type is ag::optim::SGD

// Data API
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"

namespace fs = std::filesystem;

// --------- IDX helpers ---------
static uint32_t read_be_u32(std::ifstream& ifs) {
    uint8_t b[4];
    ifs.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0])<<24) | (uint32_t(b[1])<<16) | (uint32_t(b[2])<<8) | uint32_t(b[3]);
}

struct Images {
    std::vector<uint8_t> data; // N * H * W
    uint32_t N=0, H=0, W=0;
};

static Images load_idx_images(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open " + p.string());
    if (read_be_u32(f) != 2051) throw std::runtime_error("Bad magic for images: " + p.string());
    Images I; I.N = read_be_u32(f); I.H = read_be_u32(f); I.W = read_be_u32(f);
    I.data.resize(size_t(I.N) * I.H * I.W);
    f.read(reinterpret_cast<char*>(I.data.data()), I.data.size());
    return I;
}
static std::vector<uint8_t> load_idx_labels(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open " + p.string());
    if (read_be_u32(f) != 2049) throw std::runtime_error("Bad magic for labels: " + p.string());
    uint32_t N = read_be_u32(f);
    std::vector<uint8_t> y(N);
    f.read(reinterpret_cast<char*>(y.data()), y.size());
    return y;
}

static fs::path find_file_any_depth(const fs::path& root, const std::vector<std::string>& names) {
    if (!fs::exists(root)) return {};
    for (auto& e : fs::recursive_directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto fname = e.path().filename().string();
        for (auto& n : names) if (fname == n) return e.path();
    }
    return {};
}

static std::string now_string() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// --------- Dataset (uses Example{Variable x, y}) ---------
struct MnistDataset : ag::data::Dataset {
    Images X;
    std::vector<uint8_t> y;    // class ids 0..9
    bool train = true;

    MnistDataset(const fs::path& data_root, bool train_split) : train(train_split) {
        auto p_img = find_file_any_depth(data_root, {
            train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"
        });
        auto p_lbl = find_file_any_depth(data_root, {
            train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte"
        });
        if (p_img.empty() || p_lbl.empty())
            throw std::runtime_error("MNIST files not found under " + data_root.string());

        X = load_idx_images(p_img);
        y = load_idx_labels(p_lbl);
        if (X.N != y.size()) throw std::runtime_error("MNIST N mismatch");
    }

    std::size_t size() const override { return X.N; }

    ag::data::Example get(std::size_t i) const override {
        const std::size_t img = X.H * X.W;
        std::vector<float> v; v.reserve(img);
        const std::size_t off = i * img;
        for (std::size_t j = 0; j < img; ++j) v.push_back(float(X.data[off + j]) / 255.0);

        // x shape [1, H, W]; DataLoader will stack to [B, 1, H, W]
        ag::Variable x{std::move(v), {1, X.H, X.W}, /*requires_grad=*/false};

        // y as a 1-element variable so DataLoader stacks to [B]
        ag::Variable yv{std::vector<float>{ float(y[i]) }, {1}, /*requires_grad=*/false};
        return {std::move(x), std::move(yv)};
    }
};

// Convert a batched label Variable [B] -> vector<size_t> for cross_entropy
static std::vector<std::size_t> to_index_vec(const ag::Variable& ybat) {
    const auto& v = ybat.value(); // floats
    std::vector<std::size_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<std::size_t>(v[i]);
    return out;
}

// --------- Model (Module-based; register submodules so optimizer can step(net)) ---------
struct SimpleCNN : public ag::nn::Module {
    // Submodules remain public for ad-hoc access/debug prints
    ag::nn::Conv2d   conv1{1, 8,  {3,3}, {1,1}, {1,1}};
    ag::nn::Conv2d   conv2{8, 16, {3,3}, {1,1}, {1,1}};
    ag::nn::MaxPool2d pool{2,2};
    ag::nn::Linear   fc1{16*7*7, 64};
    ag::nn::Linear   fc2{64, 10};

    // Register submodules so Module::parameters()/named_parameters() recurse into them
    SimpleCNN() {
      register_module("conv1", conv1);
      register_module("conv2", conv2);
      register_module("fc1", fc1);
      register_module("fc2", fc2);
    }

    ag::Variable forward(const ag::Variable& x) override {
        auto h = ag::relu(conv1.forward(x));
        h = pool.forward(h);
        h = ag::relu(conv2.forward(h));
        h = pool.forward(h);
        h = ag::flatten(h, 1);
        h = ag::relu(fc1.forward(h));
        return fc2.forward(h);
    }

    // No local parameters — return empty vector. Parameters live in registered submodules.
    std::vector<ag::Variable*> _parameters() override {
      return {};
    }
};

int main(int argc, char** argv) {
    fs::path data_dir = "Data";
    int epochs = 2;
    int batch = 128;
    float lr = 0.1;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto p = a.find("--data_dir=");  if (p == 0) { data_dir = a.substr(11); continue; }
        p = a.find("--epochs=");         if (p == 0) { epochs = std::stoi(a.substr(9)); continue; }
        p = a.find("--batch_size=");     if (p == 0) { batch  = std::stoi(a.substr(14)); continue; }
        p = a.find("--lr=");             if (p == 0) { lr     = std::stod(a.substr(5)); continue; }
    }

    // Datasets & loaders
    MnistDataset train_ds(data_dir, /*train=*/true);
    MnistDataset test_ds (data_dir, /*train=*/false);

    ag::data::DataLoaderOptions opts_train;
    opts_train.batch_size = batch;
    opts_train.shuffle    = true;
    opts_train.drop_last  = false;
    opts_train.seed       = 42;

    ag::data::DataLoaderOptions opts_test = opts_train;
    opts_test.shuffle = false;
    opts_test.seed    = 0;

    // either templated or non-templated; both shown:
    ag::data::DataLoader train_ld(train_ds, opts_train);
    ag::data::DataLoader test_ld (test_ds,  opts_test);

    SimpleCNN net;
    ag::nn::SGD opt(lr, /*momentum=*/0.9, /*nesterov=*/true, /*weight_decay=*/0.0);

    auto t0 = std::chrono::steady_clock::now();

    // --------- train ---------
    net.train();
    int tstep = 0;
    for (int ep = 0; ep < epochs; ++ep) {
        // train_ld.reset();  // uncomment if your loader needs an explicit reset

        float running_loss_sum = 0.0;
        std::size_t running_count = 0;

        while (train_ld.has_next()) {
            auto b = train_ld.next();                 // b.x : [B,1,28,28], b.y : [B]
            auto logits  = net.forward(b.x);
            auto targets = to_index_vec(b.y);         // std::vector<size_t>
            auto loss    = ag::nn::cross_entropy(logits, targets);
            loss.backward();

            // Debug: print param mean and grad norm for conv1 and fc2 on first 3 steps
            if (tstep < 3) {
                auto print_stats = [&](const char* name, auto& layer) {
                    auto params = layer.parameters();
                    for (std::size_t pi = 0; pi < params.size(); ++pi) {
                        const auto& val = params[pi]->value();
                        const auto& grad = params[pi]->grad();
                        float mean = 0.0f; for (float v : val) mean += v; mean /= float(val.size());
                        float gnorm = 0.0f; for (float g : grad) gnorm += g*g; gnorm = std::sqrt(gnorm);
                        std::cout << "[dbg] " << name << " param" << pi << " mean=" << mean << " grad_norm=" << gnorm << "\n";
                    }
                };
                print_stats("conv1", net.conv1);
                print_stats("fc2", net.fc2);
            }

            // single optimizer step for the entire network
            opt.step(net);

            // Debug: after step, print updated param mean for first 3 steps
            if (tstep < 3) {
                auto print_mean = [&](const char* name, auto& layer) {
                    auto params = layer.parameters();
                    for (std::size_t pi = 0; pi < params.size(); ++pi) {
                        const auto& val = params[pi]->value();
                        float mean = 0.0f; for (float v : val) mean += v; mean /= float(val.size());
                        std::cout << "[dbg] after step " << name << " param" << pi << " mean=" << mean << "\n";
                    }
                };
                print_mean("conv1", net.conv1);
                print_mean("fc2", net.fc2);
            }

            net.zero_grad();

            // update running average weighted by batch size
            const std::size_t B = targets.size();
            running_loss_sum += loss.value()[0] * float(B);
            running_count    += B;
            const float avg = running_count ? (running_loss_sum / float(running_count)) : 0.0;

            std::cout << "[train] epoch " << (ep+1) << "/" << epochs
                    << " step " << (++tstep)
                    << " avg_loss=" << std::fixed << std::setprecision(6) << avg
                    << "\n";
        }
        // train_ld.rewind(); // uncomment if your loader needs it
    }



    // --------- eval ---------
    net.eval();
    ag::NoGradGuard ng;
    std::size_t correct = 0, total = 0;

    // test_ld.reset();  // (optional; only if available)
    while (test_ld.has_next()) {
        auto b = test_ld.next();
        auto logits  = net.forward(b.x);
        auto targets = to_index_vec(b.y);
        auto loss    = ag::nn::cross_entropy(logits, targets);
        std::cout << "[test] loss=" << loss.value()[0] << "\n";

        // accuracy
        const auto& z = logits.value();
        const std::size_t B = targets.size(), C = 10;
        for (std::size_t i = 0; i < B; ++i) {
            std::size_t base = i * C, arg = 0;
            float best = z[base];
            for (std::size_t c = 1; c < C; ++c)
                if (z[base+c] > best) { best = z[base+c]; arg = c; }
            if (arg == targets[i]) ++correct;
        }
        total += B;
    }
    // test_ld.rewind();  // (optional; only if your API provides it)


    float acc = total ? (100.0 * float(correct) / float(total)) : 0.0;
    auto t1 = std::chrono::steady_clock::now();
    float seconds = std::chrono::duration<float>(t1 - t0).count();

    std::ofstream out("results_mnist.txt", std::ios::app);
    out << "Timestamp: " << now_string() << "\n"
        << "Wall time (s): " << std::fixed << std::setprecision(3) << seconds << "\n"
        << "Test accuracy (%): " << std::setprecision(2) << acc << "\n"
        << "---\n";
    out.close();

    std::cout << "Final test accuracy: " << std::setprecision(2) << acc << "%\n";
    return 0;
}
