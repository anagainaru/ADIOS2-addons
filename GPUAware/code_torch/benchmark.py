import os
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_cpu_mem_mb():
    """Returns the current process CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def log_memory(step, prefix=""):
    """Logs CPU & GPU memory usage at the current point in training."""
    cpu_mem = get_cpu_mem_mb()
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**2
        gpu_mem_reserved = torch.cuda.memory_reserved()  / 1024**2
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[{prefix} Step {step:4d}] CPU: {cpu_mem:6.1f} MB  "
              f"GPU alloc: {gpu_mem_alloc:6.1f} MB  "
              f"GPU reserved: {gpu_mem_reserved:6.1f} MB  "
              f"GPU peak alloc: {gpu_mem_max:6.1f} MB")
    else:
        print(f"[{prefix} Step {step:4d}] CPU: {cpu_mem:6.1f} MB")

# Example model & data
model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dummy dataset
data = torch.randn(5000, 1000)
targets = torch.randint(0, 10, (5000,))
dataset = TensorDataset(data, targets)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Reset peak GPU stats
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Training loop with memory logging
for epoch in range(3):
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        log_memory(step + epoch * len(loader), prefix="Before forward")
        
        out = model(x)
        loss = criterion(out, y)
        log_memory(step + epoch * len(loader), prefix="After forward")
        
        loss.backward()
        log_memory(step + epoch * len(loader), prefix="After backward")
        
        optimizer.step()
        log_memory(step + epoch * len(loader), prefix="After step")
        
    print(f"=== End of epoch {epoch} ===\n")

# Final peak usage
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory allocated over entire run: {peak:.1f} MB")
