from typing import Any
from collections.abc import Hashable
from dataclasses import dataclass


@dataclass
class HeaderInfo:
    """Extracted information from FITS header

    Attributes
    ----------
        header: dict
            header dictionary

        alias: dict
            aliases

        units: dict
            units

        comments: dict
            comments/description of keywords
    """

    header: dict[Hashable, Any]
    alias: dict[Hashable, str]
    units: dict[Hashable, str]
    comments: dict[Hashable, str]
