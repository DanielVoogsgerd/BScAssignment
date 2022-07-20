"""Custom JSON Encoder for this project that can serialize additional types."""
from json import JSONEncoder
from types import FunctionType


class CustomEncoder(JSONEncoder):
    """Custom JSON Encoder for this project that can serialize additional types."""

    def default(self, obj: object) -> str:
        """Serialize the additional types."""
        if isinstance(obj, FunctionType):
            return obj.__name__

        raise Exception("")
