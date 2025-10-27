// Additional DataLoader unit tests: exception safety, large-batch, empty dataset
#include "test_framework.hpp"
#include "ag/data/dataset.hpp"
#include "ag/data/dataloader.hpp"
#include "ag/core/variables.hpp"
#include <vector>

using ag::Variable;
using ag::data::Dataset;
using ag::data::Example;
using ag::data::InMemoryDataset;
using ag::data::DataLoader;
using ag::data::DataLoaderOptions;

static Example make_scalar_pair(float x, float y) {
    return Example{ Variable({x}, {1}, /*requires_grad=*/false),
                    Variable({y}, {1}, /*requires_grad=*/false) };
}

// Mock dataset that delegates to an internal vector but throws from get() for a
// configured index. get() is const to match the Dataset interface so we mark
// throw_idx_ mutable to allow configuration if needed.
class ThrowingDataset : public Dataset {
public:
    ThrowingDataset(std::vector<Example> items, std::size_t throw_idx)
    : items_(std::move(items)), throw_idx_(throw_idx) {}

    std::size_t size() const override { return items_.size(); }
    Example get(std::size_t idx) const override {
        if (idx == throw_idx_) throw std::runtime_error("simulated dataset failure");
        if (idx >= items_.size()) throw std::out_of_range("idx out of range");
        return items_[idx];
    }

private:
    std::vector<Example> items_;
    const std::size_t throw_idx_;
};

TEST("data/dataloader/exception_safe_cursor_advancement") {
    // Create 3-sample dataset and set batch_size=2.
    std::vector<Example> items;
    for (int i = 0; i < 3; ++i) items.push_back(make_scalar_pair(float(i), float(2*i)));

    // Construct dataset that will throw when fetching index 2 (the second batch)
    ThrowingDataset ds(std::move(items), /*throw_idx=*/2);

    DataLoaderOptions opts; opts.batch_size = 2; opts.shuffle = false; opts.drop_last = false;
    DataLoader loader(ds, opts);

    // First batch should succeed (indices 0,1)
    ASSERT_TRUE(loader.has_next());
    auto b0 = loader.next();
    ASSERT_TRUE(b0.size == 2);

    // After successful first batch, remaining should be 1
    ASSERT_TRUE(loader.remaining() == 1);

    // Second next() should attempt to fetch index 2 and throw; remaining() must stay 1
    bool threw = false;
    try {
        auto b1 = loader.next();
        (void)b1; // shouldn't reach
    } catch (const std::exception& e) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    ASSERT_TRUE(loader.remaining() == 1);
}

TEST("data/dataloader/batch_size_larger_than_dataset") {
    InMemoryDataset ds;
    for (int i = 0; i < 3; ++i) ds.push_back(make_scalar_pair(float(i), float(2*i)));

    DataLoaderOptions opts; opts.batch_size = 10; opts.shuffle = false; opts.drop_last = false;
    DataLoader loader(ds, opts);

    ASSERT_TRUE(loader.has_next());
    auto b = loader.next();
    // Should return all samples in one partial batch and then be exhausted
    ASSERT_TRUE(b.size == 3);
    ASSERT_TRUE(!loader.has_next());
}

TEST("data/dataloader/empty_dataset_behaviour") {
    InMemoryDataset ds;
    DataLoaderOptions opts; opts.batch_size = 2; opts.shuffle = false; opts.drop_last = false;
    DataLoader loader(ds, opts);

    ASSERT_TRUE(!loader.has_next());
    auto b = loader.next();
    ASSERT_TRUE(b.size == 0);
}
