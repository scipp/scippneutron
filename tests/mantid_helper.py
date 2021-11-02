def mantid_is_available():
    try:
        import mantid  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False
