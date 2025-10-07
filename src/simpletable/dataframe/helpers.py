def pretty_size_print(num_bytes: int) -> str:
    """
    Output number of bytes in a human readable format

    Parameters
    ----------
    num_bytes: int
        number of bytes to convert

    returns
    -------
    output: str
        string representation of the size with appropriate unit scale
    """
    if num_bytes is None:
        return

    KiB = 1024
    MiB = KiB * KiB
    GiB = KiB * MiB
    TiB = KiB * GiB
    PiB = KiB * TiB
    EiB = KiB * PiB
    ZiB = KiB * EiB
    YiB = KiB * ZiB

    if num_bytes > YiB:
        output = "%.3g YB" % (num_bytes / YiB)
    elif num_bytes > ZiB:
        output = "%.3g ZB" % (num_bytes / ZiB)
    elif num_bytes > EiB:
        output = "%.3g EB" % (num_bytes / EiB)
    elif num_bytes > PiB:
        output = "%.3g PB" % (num_bytes / PiB)
    elif num_bytes > TiB:
        output = "%.3g TB" % (num_bytes / TiB)
    elif num_bytes > GiB:
        output = "%.3g GB" % (num_bytes / GiB)
    elif num_bytes > MiB:
        output = "%.3g MB" % (num_bytes / MiB)
    elif num_bytes > KiB:
        output = "%.3g KB" % (num_bytes / KiB)
    else:
        output = "%.3g Bytes" % (num_bytes)

    return output
