import time
import torch

SEQ_LEN  = 4096
N_HEADS  = 10
HEAD_DIM = 96
N_RUNS   = 10
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

q      = torch.randn(N_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE)
k      = torch.randn(N_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE)
scores = torch.matmul(q, k.transpose(-2, -1)) / HEAD_DIM**0.5
mask   = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), device=DEVICE), diagonal=1)


def bench(fn, label):
    for i in range(N_RUNS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda": torch.cuda.synchronize()
        print(f"[{label}] run {i+1:2d}: {(time.perf_counter()-t0)*1000:.4f} ms")


print("\nQK^T"); bench(lambda: torch.matmul(q, k.transpose(-2, -1)), "QKt")
print("\nMask"); bench(lambda: scores + mask, "mask")


