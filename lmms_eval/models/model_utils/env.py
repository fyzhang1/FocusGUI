import os
from typing import Iterable, Optional

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None
    load_dotenv = None


_DOTENV_LOADED = False


def load_env_file() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    if load_dotenv is not None and find_dotenv is not None:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

    _DOTENV_LOADED = True


def get_env_value(
    names: Iterable[str],
    *,
    default: Optional[str] = None,
    required: bool = False,
    description: Optional[str] = None,
) -> Optional[str]:
    load_env_file()

    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value

    if required:
        joined_names = ", ".join(names)
        if description:
            raise ValueError(f"Missing {description}. Set one of: {joined_names}")
        raise ValueError(f"Missing required environment variable. Set one of: {joined_names}")

    return default
