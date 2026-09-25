import torch

def test_gpu():
    print("GPU Smoke Test")
    print("-" * 20)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Cached:    {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("WARNING: GPU not found. Running on CPU.")

if __name__ == "__main__":
    test_gpu()
