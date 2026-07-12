"""Single source of truth for the package version (ADOPT-lite campaign P5).

Before this, `version` was duplicated in pyproject.toml and
mdxnet_infer/__init__.py's `__version__`. hatchling now reads the version
from this file via `[tool.hatch.version] path =
"src/mdxnet_infer/__about__.py"` in pyproject.toml, and __init__.py imports
it from here instead of hardcoding its own copy. Bump only this file.

Reads: (nothing internal)
"""

__version__ = "0.1.0"
