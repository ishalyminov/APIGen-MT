"""Lightweight import stubs for unit tests that do not instantiate a model."""

import sys
import types

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    module = types.ModuleType("transformers")

    class _UnavailableAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("transformers is required only when loading a local tokenizer")

    module.AutoTokenizer = _UnavailableAutoTokenizer
    sys.modules["transformers"] = module
