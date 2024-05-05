import torch
from safetensors.torch import save_file


def rename(name: str):
    name = name.replace("model.0.", "net.b1.0.")
    name = name.replace("model.1.", "net.b1.1.")
    name = name.replace("model.2.m.", "net.b2.0.bottleneck.")
    name = name.replace("model.2.", "net.b2.0.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.4.m.", "net.b2.2.bottleneck.")
    name = name.replace("model.4.", "net.b2.2.")
    name = name.replace("model.5.", "net.b3.0.")
    name = name.replace("model.6.m.", "net.b3.1.bottleneck.")
    name = name.replace("model.6.", "net.b3.1.")
    name = name.replace("model.7.", "net.b4.0.")
    name = name.replace("model.8.m.", "net.b4.1.bottleneck.")
    name = name.replace("model.8.", "net.b4.1.")
    name = name.replace("model.9.", "net.b5.0.")
    name = name.replace("model.12.m.", "fpn.n1.bottleneck.")
    name = name.replace("model.12.", "fpn.n1.")
    name = name.replace("model.15.m.", "fpn.n2.bottleneck.")
    name = name.replace("model.15.", "fpn.n2.")
    name = name.replace("model.16.", "fpn.n3.")
    name = name.replace("model.18.m.", "fpn.n4.bottleneck.")
    name = name.replace("model.18.", "fpn.n4.")
    name = name.replace("model.19.", "fpn.n5.")
    name = name.replace("model.21.m.", "fpn.n6.bottleneck.")
    name = name.replace("model.21.", "fpn.n6.")
    name = name.replace("model.22.", "head.")
    return name


data = torch.load("model/yolov8n_imgsz640.pt")
print(data)
tensors = data['model'].state_dict().items()
tensors = dict(tensors)
tensors = {rename(k): t for k, t in tensors.items()}
print(data["model"])
save_file(tensors, "model/yolov8n_imgsz640.safetensors")
for k, v in tensors.items():
    print(str(k), v.shape)
