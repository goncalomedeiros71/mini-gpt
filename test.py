import torch

print("CUDA disponível?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Número de GPUs:", torch.cuda.device_count())
    print("Versão CUDA do PyTorch:", torch.version.cuda)
    print("Memória total da GPU:", torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), "GB")