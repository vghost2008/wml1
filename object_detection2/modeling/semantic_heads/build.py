from thirdparty.registry import Registry

SEMANTIC_HEAD = Registry("SEMANTIC_HEAD")

def build_semantic_head(name, *args, **kwargs):
    head = SEMANTIC_HEAD.get(name)(*args, **kwargs)
    return head