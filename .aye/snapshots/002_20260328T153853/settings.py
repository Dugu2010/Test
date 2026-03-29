"""Runtime settings dataclass."""

from dataclasses import dataclass


@dataclass
class Settings:
    model: str = "gpt-4o-mini"
    verbose: bool = False
    debug: bool = False
    theme: str = "monokai"
    max_context: int = 10
