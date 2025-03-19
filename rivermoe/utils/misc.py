def module_class_name(self, obj: type = None) -> str:
    """
    Returns the full name of the class including the module name.
    Returns self's full name if obj is None.

    Parameters
    ----------
    self : object
        The object to get the full name for.
    obj : type
        The object to get the full name for. Defaults to None.

    Returns
    -------
    str
        The full name of the class in the format 'module.ClassName'.
    """
    if obj is None:
        obj = self
    module = obj.__class__.__module__
    class_name = obj.__class__.__qualname__
    module = module.split(".")[0]
    return f"{module}.{class_name}"
