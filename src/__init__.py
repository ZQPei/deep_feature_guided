from .config import Config


from .inpainting import DeepFeatureGuided


def get_model(model):
    if model == 1:
        return DeepFeatureGuided

    else:
        raise NotImplementedError
        