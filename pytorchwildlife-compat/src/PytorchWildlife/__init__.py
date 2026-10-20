"""PytorchWildlife 2.0: a transition package that runs on Sparrow-Engine.

PytorchWildlife 1.x ran PyTorch models. 2.0 keeps the most-used 1.x entry points
(``models.detection``, ``models.classification`` and the JSON/image savers in
``utils``) and runs them on the ``sparrow_engine`` ONNX runtime with models from the
Sparrow model zoo. New code should call ``sparrow_engine`` directly; see
https://github.com/microsoft/SPARROW-Playground for notebooks.
"""

import warnings

warnings.warn(
    "PytorchWildlife 2.0 is a transition package that runs on sparrow-engine. "
    "New code should use `import sparrow_engine` directly; see "
    "https://github.com/microsoft/SPARROW-Playground.",
    DeprecationWarning,
    stacklevel=2,
)

from . import models, utils  # noqa: E402

__version__ = "2.0.0"

__all__ = ["models", "utils", "__version__"]
