// ============================
// File: tests/test_data_dataloader.cpp
// ============================
#include "test_framework.hpp"
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"
#include "ag/core/variables.hpp"
#include <vector>
#include <numeric>
#include <set>

using ag::Variable;
using ag::data::Dataset;
using ag::data::Example;
using ag::data::InMemoryDataset;
using ag::data::DataLoader;
using ag::data::DataLoaderOptions;

static Example make_scalar_pair(double x, double y) {
    return Example{ Variable({x}, {1}, /*requires_grad=*/false),
                    Variable({y}, {1}, /*requires_grad=*/false) };
}

TEST("data/dataloader/basic_no_shuffle_drop_last_false") {
    // Build N=5 samples with x=i, y=2*i
    InMemoryDataset ds;
    for (int i=0;i<5;++i) ds.push_back(make_scalar_pair(i, 2*i));

    DataLoaderOptions opts;
    opts.batch_size = 2;
    opts.shuffle = false;
    opts.drop_last = false;
    DataLoader loader(ds, opts);

    std::vector<double> seen_x;
    std::vector<std::size_t> batch_sizes;
    while (loader.has_next()) {
        auto b = loader.next();
        batch_sizes.push_back(b.size);
        for (double v : b.x.value()) seen_x.push_back(v); // flat order mirrors dataset
    }

    ASSERT_TRUE(batch_sizes.size() == 3);               // [2,2,1]
    ASSERT_TRUE(batch_sizes[0] == 2 && batch_sizes[1] == 2 && batch_sizes[2] == 1);
    ASSERT_TRUE(seen_x.size() == 5);
    for (int i=0;i<5;++i) ASSERT_NEAR(seen_x[i], (double)i, 1e-12);
}

TEST("data/dataloader/drop_last_true") {
    InMemoryDataset ds;
    for (int i=0;i<5;++i) ds.push_back(make_scalar_pair(i, 2*i));

    DataLoaderOptions opts;
    opts.batch_size = 2;
    opts.shuffle = false;
    opts.drop_last = true;
    DataLoader loader(ds, opts);

    std::size_t total = 0, batches = 0;
    while (loader.has_next()) {
        auto b = loader.next();
        total += b.size;
        ++batches;
    }
    ASSERT_TRUE(batches == 2);
    ASSERT_TRUE(total == 4); // last partial batch dropped
}

TEST("data/dataloader/shuffle_seeded_reproducible") {
    InMemoryDataset ds;
    const int N = 10;
    for (int i=0;i<N;++i) ds.push_back(make_scalar_pair(i, 2*i));

    DataLoaderOptions opts;
    opts.batch_size = 3;
    opts.shuffle = true;
    opts.drop_last = false;
    opts.seed = 42ull;

    DataLoader loader1(ds, opts);
    std::vector<double> order1;
    while (loader1.has_next()) {
        auto b = loader1.next();
        for (double v : b.x.value()) order1.push_back(v);
    }

    // Second loader with the same seed should reproduce order
    DataLoader loader2(ds, opts);
    std::vector<double> order2;
    while (loader2.has_next()) {
        auto b = loader2.next();
        for (double v : b.x.value()) order2.push_back(v);
    }

    ASSERT_TRUE(order1.size() == order2.size());
    for (std::size_t i=0;i<order1.size();++i) ASSERT_NEAR(order1[i], order2[i], 1e-12);

    // And it should be a permutation (no repeats, all present)
    std::multiset<int> ms;
    for (double v : order1) ms.insert((int)std::llround(v));
    ASSERT_TRUE(ms.size() == (std::size_t)N);
    for (int i=0;i<N;++i) ASSERT_TRUE(ms.count(i) == 1);
}
