from .spec import ModelSpec
from .low_noise_refiner import AlphaGateRefiner, LowNoiseAdaptiveRefiner
from .vm import VelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'LowNoiseAdaptiveRefiner': LowNoiseAdaptiveRefiner,
        'AlphaGateRefiner': AlphaGateRefiner,
        'VelocityModule': VelocityModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
