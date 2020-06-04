from thirdparty.registry import Registry

MATCHER = Registry("MATCHER")  # noqa F401 isort:skip


def build_matcher(name,*args,**kwargs):
    return MATCHER.get(name)(*args,**kwargs)

