// filepath: src/ag/nn/layers/upsample.cpp
#include "ag/nn/layers/upsample.hpp"
#include "ag/ops/tensor_utils.hpp"
#include "ag/parallel/parallel_for.hpp"
#include <stdexcept>

namespace ag {
using detail::numel;

Variable upsample2d(const Variable& x, std::size_t scale_h, std::size_t scale_w) {
  const auto& xs = x.n->shape;
  if (xs.size() != 4)
    throw std::invalid_argument("upsample2d: input must be [B,C,H,W]");
  if (scale_h == 0 || scale_w == 0)
    throw std::invalid_argument("upsample2d: scale factors must be >= 1");

  const std::size_t B = xs[0], C = xs[1], H = xs[2], W = xs[3];
  const std::size_t H_out = H * scale_h;
  const std::size_t W_out = W * scale_w;
  const std::vector<std::size_t> out_shape = {B, C, H_out, W_out};
  const std::size_t out_numel = B * C * H_out * W_out;

  auto out = std::make_shared<Node>();
  out->shape = out_shape;
  out->value.resize(out_numel);
  out->grad.assign(out_numel, 0.0f);
  out->requires_grad = x.n->requires_grad;
  out->parents = {x.n};

  // Forward: nearest-neighbor repeat.
  // Parallelize over (B*C) channels — each channel is independent.
  const std::size_t BC = B * C;
  const float* xdata = x.n->value.data();
  float* odata = out->value.data();

  ag::parallel::parallel_for(BC, /*grain=*/1, [&](std::size_t bc0, std::size_t bc1) {
    for (std::size_t bc = bc0; bc < bc1; ++bc) {
      const float* in_plane = xdata + bc * H * W;
      float* out_plane = odata + bc * H_out * W_out;
      for (std::size_t oh = 0; oh < H_out; ++oh) {
        const std::size_t ih = oh / scale_h;
        for (std::size_t ow = 0; ow < W_out; ++ow) {
          const std::size_t iw = ow / scale_w;
          out_plane[oh * W_out + ow] = in_plane[ih * W + iw];
        }
      }
    }
  });

  // Backward: accumulate output grad into input grad by mapping each output pixel
  // back to its source pixel.
  out->backward = [Xn = x.n, xs_cap = xs, out_shape,
                   scale_h, scale_w,
                   oweak = std::weak_ptr<ag::Node>(out)]() {
    auto op = oweak.lock();
    if (!op) return;
    if (!Xn || !Xn->requires_grad) return;

    const std::size_t B = xs_cap[0], C = xs_cap[1], H = xs_cap[2], W = xs_cap[3];
    const std::size_t H_out = H * scale_h;
    const std::size_t W_out = W * scale_w;

    if (Xn->grad.size() != Xn->value.size())
      Xn->grad.assign(Xn->value.size(), 0.0f);

    const float* grad_out = op->grad.data();
    float* grad_in = Xn->grad.data();
    const std::size_t BC = B * C;

    ag::parallel::parallel_for(BC, /*grain=*/1, [&](std::size_t bc0, std::size_t bc1) {
      for (std::size_t bc = bc0; bc < bc1; ++bc) {
        const float* go = grad_out + bc * H_out * W_out;
        float* gi = grad_in + bc * H * W;
        for (std::size_t oh = 0; oh < H_out; ++oh) {
          const std::size_t ih = oh / scale_h;
          for (std::size_t ow = 0; ow < W_out; ++ow) {
            const std::size_t iw = ow / scale_w;
            gi[ih * W + iw] += go[oh * W_out + ow];
          }
        }
      }
    });
  };

  return make_from_node(out);
}

} // namespace ag
