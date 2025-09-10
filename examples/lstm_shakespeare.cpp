// ============================
// File: examples/lstm_shakespeare.cpp
// ============================
// Tiny Shakespeare LSTM demo for the AutoCalc stack.
// - Loads UTF-8 text from --data_path (default Data/TinyShakespeare/input.txt)
// - Builds a char-level dataset with fixed-length windows
// - Trains a 2-layer LSTM (H=256, dropout=0.2) + Linear head
// - Prints per-batch train loss (and throughput) and per-epoch validation metrics
// - Appends Timestamp, Wall time (s), and Valid accuracy (%) to results_lstm.txt
//
// This example mirrors the style of examples/mnist_demo.cpp (CLI, logs, optimizer usage).
// If naming in your codebase differs slightly (e.g., DataLoader API), adjust those lines accordingly.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ag/core/variables.hpp"
#include "ag/nn/module.hpp"
#include "ag/nn/layers/lstm.hpp"
#include "ag/nn/layers/linear.hpp"
#include "ag/nn/dropout.hpp"
#include "ag/nn/loss.hpp"
#include "ag/nn/optim/sgd.hpp"
#include "ag/ops/reshape.hpp"
#include "ag/ops/activations.hpp"
#include "ag/ops/graph.hpp"          // NoGradGuard
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"

namespace fs = std::filesystem;

// ---------- tiny CLI helper ----------
static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defval) {
    const std::string pref = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind(pref, 0) == 0) return s.substr(pref.size());
    }
    return defval;
}

static int get_arg_int(int argc, char** argv, const std::string& key, int defval) {
    try { return std::stoi(get_arg(argc, argv, key, std::to_string(defval))); }
    catch (...) { return defval; }
}

static double get_arg_double(int argc, char** argv, const std::string& key, double defval) {
    try { return std::stod(get_arg(argc, argv, key, std::to_string(defval))); }
    catch (...) { return defval; }
}

// ---------- tiny time helper ----------
struct ScopedTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    ScopedTimer() : t0(clock::now()) {}
    double elapsed_s() const { 
        using namespace std::chrono;
        return duration_cast<duration<double>>(clock::now() - t0).count(); 
    }
};

// ---------- dataset: char-level fixed windows ----------
struct TinyShakespeareDataset : ag::data::Dataset {
    // Each sample: x_onehot [T,V] (double), y_ids [T] (double IDs for batching)
    std::vector<std::vector<double>> xs;   // flattened T*V one-hot
    std::vector<std::vector<double>> ys;   // length T (double IDs)
    std::size_t T;
    std::size_t V;

    TinyShakespeareDataset(const std::vector<int>& ids, std::size_t vocab_size, std::size_t seq_len, std::size_t start, std::size_t end_inclusive)
    : T(seq_len), V(vocab_size)
    {
        if (end_inclusive <= start || end_inclusive >= ids.size()) throw std::runtime_error("bad dataset range");
        // Non-overlapping windows of length T
        const std::size_t last = end_inclusive - T - 1; // we access i..i+T and i+1..i+T
        for (std::size_t i = start; i <= last; i += T) {
            std::vector<double> x; x.resize(T * V, 0.0);
            std::vector<double> y; y.resize(T, 0.0);
            for (std::size_t t = 0; t < T; ++t) {
                int in_id = ids[i + t];
                int out_id = ids[i + t + 1];
                // one-hot at [t, in_id]
                x[t * V + in_id] = 1.0;
                y[t] = static_cast<double>(out_id);
            }
            xs.emplace_back(std::move(x));
            ys.emplace_back(std::move(y));
        }
    }

    std::size_t size() const override { return xs.size(); }

    ag::data::Example get(std::size_t idx) const override {
        const auto& x = xs[idx];
        const auto& y = ys[idx];
        ag::Variable vx(x, {T, V}, /*requires_grad=*/false);
        ag::Variable vy(y, {T}, /*requires_grad=*/false); // double-encoded ids; we'll cast to size_t later
        return {vx, vy};
    }
};

// ---------- utility: read whole file ----------
static std::string read_text(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("failed to open " + path);
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return s;
}

// ---------- build vocab (char -> id) ----------
struct Vocab {
    std::unordered_map<char, int> to_id;
    std::vector<char> to_char;
};

static Vocab build_vocab(const std::string& text) {
    std::unordered_set<char> uniq(text.begin(), text.end());
    std::vector<char> chars(uniq.begin(), uniq.end());
    std::sort(chars.begin(), chars.end());
    Vocab v;
    v.to_char = chars;
    for (int i = 0; i < (int)chars.size(); ++i) v.to_id[chars[i]] = i;
    return v;
}

static std::vector<int> encode_ids(const std::string& text, const Vocab& v) {
    std::vector<int> ids; ids.reserve(text.size());
    for (char c : text) {
        auto it = v.to_id.find(c);
        if (it == v.to_id.end()) throw std::runtime_error("unknown char during encoding");
        ids.push_back(it->second);
    }
    return ids;
}

// ---------- convert batched target Variable [B,T] (double IDs) -> vector<size_t> length B*T ----------
static std::vector<std::size_t> to_index_vec_flatten_BT(const ag::Variable& yBT) {
    const auto& shp = yBT.shape();
    if (shp.size() != 2) throw std::runtime_error("expected [B,T] target shape");
    const std::size_t B = shp[0], T = shp[1];
    const auto& vals = yBT.value();
    if (vals.size() != B * T) throw std::runtime_error("bad target size");
    std::vector<std::size_t> out; out.reserve(B * T);
    for (std::size_t i = 0; i < B * T; ++i) {
        // stored as double ids in [0, V-1]
        double d = vals[i];
        if (d < 0) d = 0;
        out.push_back(static_cast<std::size_t>(d));
    }
    return out;
}

// ---------- argmax over last dim of logits [N, V] ----------
static std::vector<std::size_t> argmax_rows(const ag::Variable& logits) {
    const auto& shp = logits.shape();
    if (shp.size() != 2) throw std::runtime_error("expected [N,V] logits");
    const std::size_t N = shp[0], V = shp[1];
    const auto& vals = logits.value();
    std::vector<std::size_t> pred(N, 0);
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t best = 0;
        double bestv = vals[i * V + 0];
        for (std::size_t j = 1; j < V; ++j) {
            double v = vals[i * V + j];
            if (v > bestv) { bestv = v; best = j; }
        }
        pred[i] = best;
    }
    return pred;
}

static double accuracy_from_preds(const std::vector<std::size_t>& pred, const std::vector<std::size_t>& tgt) {
    if (pred.size() != tgt.size()) throw std::runtime_error("acc size mismatch");
    std::size_t correct = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) if (pred[i] == tgt[i]) ++correct;
    return 100.0 * double(correct) / double(pred.size());
}

// ---------- model ----------
struct LSTMNet : ag::nn::Module {
    ag::nn::LSTM lstm;        // expects [B,T,V] -> [B,T,H]
    ag::nn::Dropout drop_between;
    ag::nn::Linear fc;        // [H] -> [V]
    std::size_t H;
    std::size_t V;

    LSTMNet(std::size_t input_dim_V, std::size_t hidden_H, int layers_L, double dropout_p)
    : lstm(/*input_size=*/input_dim_V, /*hidden_size=*/hidden_H, /*num_layers=*/layers_L, /*bias=*/true),
      drop_between(dropout_p),
      fc(hidden_H, input_dim_V, /*bias=*/true),
      H(hidden_H), V(input_dim_V)
    {
        register_module("lstm", lstm);
        register_module("drop", drop_between);
        register_module("fc",   fc);
    }

    ag::Variable forward(const ag::Variable& xBTv) override {
        // xBTv: [B,T,V]
        auto hBTk = lstm.forward(xBTv);                 // [B,T,H]
        if (training()) hBTk = drop_between.forward(hBTk);
        // reshape to [B*T, H]
        const auto& shp = hBTk.shape();
        std::size_t B = shp[0], T = shp[1];
        auto flat = ag::reshape(hBTk, {B * T, H});      // [B*T, H]
        auto logits = fc.forward(flat);                 // [B*T, V]
        return logits;
    }

    std::vector<ag::Variable*> _parameters() override {
        // Parameters live in submodules; return empty here
        return {};
    }
};

// ---------- training / validation ----------
int main(int argc, char** argv) {
    // CLI
    const std::string data_path = get_arg(argc, argv, "data_path", "Data/TinyShakespeare/input.txt");
    const int    EPOCHS    = get_arg_int(argc, argv, "epochs",    4);
    const int    BATCH     = get_arg_int(argc, argv, "batch_size", 256);
    const int    T         = get_arg_int(argc, argv, "seq_len",   128);
    const int    HIDDEN    = get_arg_int(argc, argv, "hidden",    256);
    const int    LAYERS    = get_arg_int(argc, argv, "layers",    2);
    const double DROPOUT_P = get_arg_double(argc, argv, "dropout", 0.2);
    const double LR        = get_arg_double(argc, argv, "lr",      0.05);
    const double MOM       = get_arg_double(argc, argv, "momentum",0.9);
    const int    LOG_INT   = get_arg_int(argc, argv, "log_interval", 50);

    // Load and encode data
    std::cerr << "Loading text from: " << data_path << "\n";
    const std::string text = read_text(data_path);
    const auto vocab = build_vocab(text);
    const auto ids   = encode_ids(text, vocab);
    const std::size_t V = vocab.to_char.size();
    if (V < 10) std::cerr << "Warning: very small vocab (" << V << ")\n";

    // Split 90/10 contiguously
    const std::size_t N = ids.size();
    const std::size_t split = (N * 90) / 100;
    const std::size_t train_end = split - 1;
    const std::size_t valid_end = N - 2; // we access +1

    // Build datasets
    TinyShakespeareDataset train_ds(ids, V, T, /*start=*/0,            /*end_inclusive=*/train_end);
    TinyShakespeareDataset valid_ds(ids, V, T, /*start=*/split,        /*end_inclusive=*/valid_end);

    // DataLoaders
    ag::data::DataLoaderOptions topts;
    topts.batch_size = std::size_t(BATCH);
    topts.shuffle    = true;
    topts.drop_last  = true;
    topts.seed       = 42;
    ag::data::DataLoader train_ld(train_ds, topts);

    ag::data::DataLoaderOptions vopts = topts;
    vopts.shuffle = false;
    ag::data::DataLoader valid_ld(valid_ds, vopts);

    // Model + Optimizer
    LSTMNet net(/*input_dim_V=*/V, /*hidden_H=*/HIDDEN, /*layers_L=*/LAYERS, /*dropout_p=*/DROPOUT_P);
    ag::nn::SGD opt(LR, MOM, /*nesterov=*/true, /*weight_decay=*/5e-4);
    std::cerr << "Begun Training...\n";
    // Training
    double wall_time_s = 0.0;
    auto wall = ScopedTimer();
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        net.train();
        train_ld.reset(); // if your loader needs it; otherwise it's a no-op
        std::size_t step = 0;
        while (train_ld.has_next()) {
            auto step_timer = ScopedTimer();
            auto b = train_ld.next();            // b.x [B,T,V], b.y [B,T] (double IDs)
            auto logits = net.forward(b.x);      // [B*T,V]
            auto targets = to_index_vec_flatten_BT(b.y); // [B*T] size_t
            auto loss = ag::nn::cross_entropy(logits, targets);

            loss.backward();
            opt.step(net.lstm);
            opt.step(net.fc);
            net.lstm.zero_grad();
            net.fc.zero_grad();

            const double step_s = step_timer.elapsed_s();
            const double throughput = (double(BATCH) * double(T)) / std::max(1e-9, step_s);
            ++step;
            std::cout << "[train] epoch " << epoch << "/" << EPOCHS
                        << " step " << step
                        << " loss=" << loss.value()[0]
                        << " throughput=" << throughput << " chars/s\n";
            
        }

        // Validation
        net.eval();
        ag::NoGradGuard ng;
        valid_ld.reset();
        double total_loss = 0.0;
        std::size_t total_tokens = 0;
        std::size_t step_v = 0;
        double total_eval_time = 0.0;
        while (valid_ld.has_next()) {
            auto st = ScopedTimer();
            auto b = valid_ld.next();
            auto logits = net.forward(b.x);                 // [B*T,V]
            auto targets = to_index_vec_flatten_BT(b.y);    // [B*T]
            auto loss = ag::nn::cross_entropy(logits, targets);
            total_loss += loss.value()[0] * targets.size();
            total_tokens += targets.size();

            auto pred = argmax_rows(logits);
            double acc = accuracy_from_preds(pred, targets);
            total_eval_time += st.elapsed_s();

            // Print one line per valid batch loss (optional, uncomment if desired)
            std::cout << "[valid-batch] loss=" << loss.value()[0] << " acc=" << acc << "%\n";
            ++step_v;
        }
        const double mean_loss = total_tokens ? (total_loss / double(total_tokens)) : 0.0;
        const double ppl = std::exp(mean_loss);
        const double throughput_valid = total_tokens / std::max(1e-9, total_eval_time);

        std::cout << "[valid] epoch " << epoch
                  << " loss=" << mean_loss
                  << " ppl=" << ppl
                  << " acc=" << "(per-batch shown if enabled)"
                  << " throughput=" << throughput_valid << " chars/s\n";
    }
    wall_time_s = wall.elapsed_s();

    // Final validation accuracy (single pass) for the log file
    net.eval();
    ag::NoGradGuard ng2;
    valid_ld.reset();
    double total_loss2 = 0.0;
    std::size_t total_tokens2 = 0;
    std::size_t correct2 = 0;
    while (valid_ld.has_next()) {
        auto b = valid_ld.next();
        auto logits = net.forward(b.x);
        auto targets = to_index_vec_flatten_BT(b.y);
        auto pred = argmax_rows(logits);
        // compute accuracy
        for (std::size_t i = 0; i < pred.size(); ++i) {
            if (pred[i] == targets[i]) ++correct2;
        }
        total_tokens2 += targets.size();
    }
    const double final_acc = total_tokens2 ? (100.0 * double(correct2) / double(total_tokens2)) : 0.0;

    // Append concise results to results_lstm.txt (like results_mnist.txt style)
    const std::string out_path = "results_lstm.txt";
    std::ofstream ofs(out_path, std::ios::app);
    if (ofs) {
        // Local timestamp
        std::time_t t = std::time(nullptr);
        char buf[64]; std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        ofs << "Timestamp: " << buf << "\n";
        ofs << "Wall time (s): " << wall_time_s << "\n";
        ofs << "Valid accuracy (%): " << final_acc << "\n";
        ofs << "------------------------------\n";
        ofs.close();
        std::cout << "Appended results to " << out_path << "\n";
    } else {
        std::cerr << "Failed to open " << out_path << " for writing.\n";
    }

    std::cout << "Final valid accuracy: " << final_acc << "%\n";
    return 0;
}
