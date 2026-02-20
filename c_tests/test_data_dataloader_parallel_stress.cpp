// #include "test_framework.hpp"
// #include "ag/data/dataset.hpp"
// #include "ag/data/dataloader.hpp"
// #include "ag/core/variables.hpp"
// #include <vector>

// using ag::Variable;
// using ag::data::Dataset;
// using ag::data::Example;
// using ag::data::InMemoryDataset;
// using ag::data::DataLoader;
// using ag::data::DataLoaderOptions;

// static Example make_scalar(std::size_t i) {
//     float x = static_cast<float>(i % 10);
//     return Example{ Variable({x}, {1}, /*requires_grad=*/false),
//                     Variable({x}, {1}, /*requires_grad=*/false) };
// }

// TEST("data/dataloader/parallel_prefetch_epoch_stops") {
//     const std::size_t num_samples = 512;
//     const std::size_t batch_size  = 64;
//     const std::size_t num_workers = 4;
//     const std::size_t prefetch    = 8;

//     InMemoryDataset ds;
//     for (std::size_t i = 0; i < num_samples; ++i) ds.push_back(make_scalar(i));

//     DataLoaderOptions opts;
//     opts.batch_size = batch_size;
//     opts.shuffle = true;
//     opts.drop_last = false;
//     opts.seed = 1234ull;
//     opts.num_workers = num_workers;
//     opts.prefetch_batches = prefetch;

//     DataLoader loader(ds, opts);

//     const std::size_t expected_batches = (num_samples + batch_size - 1) / batch_size;
//     const std::size_t iter_cap = expected_batches + 2; // detect runaway

//     // Iterate one epoch; guard with iter_cap to avoid hangs if backend loops.
//     std::size_t n_batches = 0;
//     for (std::size_t i = 0; i < iter_cap && loader.has_next(); ++i) {
//         auto b = loader.next();
//         if (b.size == 0) break; // end signaled defensively
//         ++n_batches;
//     }
//     ASSERT_EQ(n_batches, expected_batches);

//     // Rewind and iterate again to exercise epoch++ shuffle reseed
//     loader.rewind();
//     std::size_t n2 = 0;
//     for (std::size_t i = 0; i < iter_cap && loader.has_next(); ++i) {
//         auto b = loader.next();
//         if (b.size == 0) break;
//         ++n2;
//     }
//     ASSERT_EQ(n2, expected_batches);
// }
