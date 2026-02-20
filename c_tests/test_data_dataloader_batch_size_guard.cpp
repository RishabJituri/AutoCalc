#include "test_framework.hpp"
#include "ag/data/dataloader.hpp"
#include "ag/data/dataset.hpp"

using ag::data::InMemoryDataset;
using ag::data::DataLoaderOptions;
using ag::data::DataLoader;
using ag::data::Example;
using ag::Variable;

static Example make_scalar_pair(float x, float y) {
    return Example{ Variable({x}, {1}, /*requires_grad=*/false), Variable({y}, {1}, /*requires_grad=*/false) };
}

TEST("data/dataloader/invalid_batch_size_throws_explicit") {
    InMemoryDataset ds; ds.push_back(make_scalar_pair(0.0f, 0.0f));
    DataLoaderOptions opts; opts.batch_size = 0; opts.shuffle = false;
    ASSERT_THROWS(DataLoader(ds, opts));
}
