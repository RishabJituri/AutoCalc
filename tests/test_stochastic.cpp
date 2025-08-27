
// tests/test_stochastic.cpp
#include "test_framework.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

#include "ag/core/stochastic.hpp"
#include "ag/core/variables.hpp"

using ag::Variable;
using ag::CategoricalSample;
using ag::GumbelSoftmax;

// Softmax helper for plain doubles
static std::vector<double> softmax(const std::vector<double>& logits) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : logits) if (x > m) m = x;
    double Z = 0.0;
    for (double x : logits) Z += std::exp(x - m);
    std::vector<double> p(logits.size());
    for (size_t i=0;i<logits.size();++i) p[i] = std::exp(logits[i]-m)/Z;
    return p;
}

TEST("stochastic/categorical_frequencies_match_softmax") {
    std::vector<double> ll = {0.1, -0.2, 1.3, 0.0};
    Variable logits(ll, {ll.size()}, /*requires_grad=*/false);
    const int N = 20000;
    std::vector<int> counts(ll.size(), 0);
    for (int t=0;t<N;++t) {
        auto out = CategoricalSample(logits, 123456789ull + t);
        // out.sample_onehot is [K]; find argmax
        size_t arg = 0; double best=-1e9;
        for (size_t k=0;k<ll.size();++k) {
            double v = out.sample_onehot.value()[k];
            if (v > best) { best=v; arg=k; }
        }
        counts[arg]++;
    }
    auto p = softmax(ll);
    for (size_t k=0;k<p.size();++k) {
        double freq = counts[k] / double(N);
        ASSERT_NEAR(freq, p[k], 0.02);
    }
}

TEST("stochastic/gumbel_softmax_low_tau_is_peaky") {
    std::vector<double> ll = {2.0, 0.0, -1.0};
    Variable logits(ll, {ll.size()}, /*requires_grad=*/false);
    auto y = GumbelSoftmax(logits, /*tau=*/0.1, /*hard=*/false, /*seed=*/987654321ull);
    double sum = 0.0, maxv = -1e9;
    for (double v : y.value()) { sum += v; if (v > maxv) maxv = v; }
    ASSERT_NEAR(sum, 1.0, 1e-6);
    ASSERT_TRUE(maxv > 0.9);
}

// --- Added tests: logits shift invariance, logprob check, temp extremes, reproducibility ---

static std::vector<double> logsoftmax(const std::vector<double>& logits) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : logits) if (x > m) m = x;
    double Z = 0.0;
    for (double x : logits) Z += std::exp(x - m);
    double lse = std::log(Z) + m;
    std::vector<double> ls(logits.size());
    for (size_t i=0;i<logits.size();++i) ls[i] = logits[i] - lse;
    return ls;
}

TEST("stochastic/categorical_logits_shift_invariance") {
    std::vector<double> base = {0.5, -1.2, 2.3, 0.0};
    std::vector<double> shifted = {0.5+3.7, -1.2+3.7, 2.3+3.7, 0.0+3.7};
    const int N = 20000;
    std::vector<int> c1(base.size(),0), c2(base.size(),0);
    Variable L1(base, {base.size()}, false);
    Variable L2(shifted, {shifted.size()}, false);
    for (int t=0;t<N;++t) {
        auto s1 = CategoricalSample(L1, 424242ull + t);
        auto s2 = CategoricalSample(L2, 424242ull + t);
        size_t i1=0, i2=0; double b1=-1e9, b2=-1e9;
        for (size_t k=0;k<base.size();++k) {
            if (s1.sample_onehot.value()[k] > b1) { b1=s1.sample_onehot.value()[k]; i1=k; }
            if (s2.sample_onehot.value()[k] > b2) { b2=s2.sample_onehot.value()[k]; i2=k; }
        }
        c1[i1]++; c2[i2]++;
    }
    for (size_t k=0;k<base.size();++k) {
        double f1 = c1[k] / double(N), f2 = c2[k] / double(N);
        ASSERT_NEAR(f1, f2, 0.02);
    }
}

TEST("stochastic/categorical_logprob_matches_choice") {
    std::vector<double> l = {0.3, 1.1, -0.7};
    Variable logits(l, {l.size()}, false);
    auto out = CategoricalSample(logits, 123u);
    size_t idx=0; double best=-1e9;
    for (size_t k=0;k<l.size();++k) {
        if (out.sample_onehot.value()[k] > best) { best = out.sample_onehot.value()[k]; idx = k; }
    }
    auto ls = logsoftmax(l);
    ASSERT_NEAR(out.logprob.value()[0], ls[idx], 1e-8);
    double s=0; for (double v : out.sample_onehot.value()) s+=v;
    ASSERT_NEAR(s, 1.0, 1e-12);
}

TEST("stochastic/gumbel_softmax_temperature_extremes") {
    std::vector<double> l = {1.0, 0.0, -2.0};
    Variable logits(l, {l.size()}, false);
    // very low tau -> nearly one-hot
    auto low = GumbelSoftmax(logits, 0.05, false, 77ull);
    double s1=0, max1=-1e9; for (double v : low.value()){ s1+=v; if (v>max1) max1=v; }
    ASSERT_NEAR(s1, 1.0, 1e-6);
    ASSERT_TRUE(max1 > 0.97);
    // very high tau -> ~uniform
    auto high = GumbelSoftmax(logits, 50.0, false, 77ull);
    double s2=0; for (double v : high.value()) s2+=v;
    ASSERT_NEAR(s2, 1.0, 1e-6);
    for (double v : high.value()) ASSERT_NEAR(v, 1.0/3.0, 0.05);
}

TEST("stochastic/reproducible_with_same_seed") {
    std::vector<double> l = {0.2, -0.3, 0.7, 1.0};
    Variable logits(l, {l.size()}, false);
    auto a = GumbelSoftmax(logits, 0.5, false, 999ull);
    auto b = GumbelSoftmax(logits, 0.5, false, 999ull);
    ASSERT_TRUE(a.value().size() == b.value().size());
    for (size_t i=0;i<a.value().size();++i) ASSERT_NEAR(a.value()[i], b.value()[i], 0.0);
}
