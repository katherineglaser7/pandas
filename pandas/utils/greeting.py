def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Parameters
    ----------
    name : str
        The name to greet.

    Returns
    -------
    str
        A greeting message in the format "Hello, {name}!".

    Examples
    --------
    >>> greet('World')
    'Hello, World!'
    """
    return f"Hello, {name}!"
