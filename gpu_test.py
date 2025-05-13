import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 첫 번째 GPU 이름 출력
