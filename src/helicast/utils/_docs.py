from functools import partial

__all__ = ["link_docs_to_class"]


def link_docs_to_class(func=None, *, cls):
    if func is None:
        return partial(link_docs_to_class, cls=cls)

    if cls.__doc__ is None:
        return func

    docs = cls.__doc__.split("\n")
    for i, line in enumerate(docs):
        if line.strip() == "Args:":
            break
    docs.insert(i + 1, "        X: Input DataFrame")

    path = cls.__module__.split(".") + [cls.__name__]
    path = [i for i in path if not i.startswith("_")]
    path = ".".join(path)
    docs += ["", f"See also :class:`{path}` (class API)."]

    for i in range(len(docs)):
        if docs[i].startswith("    "):
            docs[i] = docs[i][4:]

    func.__doc__ = "\n".join(docs)

    docs = cls.__doc__

    path = func.__module__.split(".") + [func.__name__]
    path = [i for i in path if not i.startswith("_")]
    path = ".".join(path)
    docs += f"\n\nSee also :class:`{path}` (functional API)."
    docs = docs.split("\n")
    for i in range(len(docs)):
        if docs[i].startswith("    "):
            docs[i] = docs[i][4:]
    docs = "\n".join(docs)
    cls.__doc__ = docs

    return func
