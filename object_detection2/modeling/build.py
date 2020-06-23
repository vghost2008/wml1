from thirdparty.registry import Registry

HEAD_OUTPUTS = Registry("HEAD_OUTPUTS")

def build_outputs(name, *args, **kwargs):
    outputs = HEAD_OUTPUTS.get(name)(*args, **kwargs)
    return outputs