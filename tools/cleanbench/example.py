"""Example: time a batch-1 fp16 GEMV and an elementwise kernel with cleanbench (cold and steady)."""
import torch
import torch.nn.functional as F

import cleanbench as cb

l2 = torch.cuda.get_device_properties(0).L2_cache_size
W = (torch.randn(14336, 4096, device="cuda") * 0.02).half()
x = torch.randn(1, 4096, device="cuda").half()
print("GEMV cold  :", round(cb.bench(lambda: F.linear(x, W)).median_us, 2), "us")

n = max(2, -(-4 * l2 // (2 * (l2 // 4))))                     # copies covering >= 4 x L2
xs = [torch.randn(l2 // 16, device="cuda") for _ in range(n)]  # L2/4 bytes each
ys = [torch.empty_like(t) for t in xs]
cold = cb.bench(lambda: torch.mul(xs[0], 2.0, out=ys[0]))
steady = cb.bench(rotate=[lambda i=i: torch.mul(xs[i], 2.0, out=ys[i]) for i in range(n)], mode="steady")
print("elementwise cold / steady:", round(cold.median_us, 2), "/", round(steady.median_us, 2), "us", cold.warnings)
