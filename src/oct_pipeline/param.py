
def parameters_info(model):
    for name, param in model.named_parameters():
        print(f"{name:50} | Shape: {str(param.shape):20} | Trainable: {param.requires_grad}")
    