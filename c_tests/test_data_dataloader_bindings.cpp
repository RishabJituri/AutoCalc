// #include <iostream>
// #include <memory>
// #include "ag/data/dataloader.hpp"
// #include "ag/data/dataset.hpp"

// using namespace ag::data;

// struct SimpleDataset : public Dataset {
//   SimpleDataset(size_t n): n_(n){}
//   std::size_t size() const override { return n_; }
//   Example get(std::size_t idx) const override {
//     Example e;
//     std::vector<float> v(1, static_cast<float>(idx));
//     e.x = ag::Variable(v, std::vector<std::size_t>{1}, false);
//     e.y = ag::Variable(std::vector<float>{float(idx%10)}, std::vector<std::size_t>{1}, false);
//     return e;
//   }
//   size_t n_;
// };

// int main(){
//   SimpleDataset ds(10);
//   auto loader = DataLoader(std::make_shared<SimpleDataset>(ds), DataLoaderOptions{3,false,false,0});
//   size_t count = 0;
//   while(loader.has_next()){
//     auto b = loader.next();
//     std::cout<<"batch size="<<b.size<<" first="<<b.x.value()[0]<<"\n";
//     ++count;
//   }
//   std::cout<<"batches="<<count<<"\n";
//   return 0;
// }
