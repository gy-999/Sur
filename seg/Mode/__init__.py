from seg.Mode.model import *
MODELS = {
    "EA_Unet": EA_Unet,
}

def get_model(model_name, in_channels=4, out_channels=4):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name](in_channels, out_channels)