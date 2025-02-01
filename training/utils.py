import mlx.core as mx

def calculate_loss(predicted, target):
    return mx.mean((predicted - target) ** 2)
