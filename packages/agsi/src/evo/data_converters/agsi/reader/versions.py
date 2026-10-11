#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""AGSi schema version detection. Only AGSi 1.0.x documents are supported."""

import re
from typing import Any, Optional

from .errors import AGSIUnsupportedVersionError

SUPPORTED_VERSIONS = "1.0.x"
# A full "1.0.<patch>" version: no bare "1.0", extra components, prerelease/build suffixes,
# leading zeros or surrounding whitespace.
_SUPPORTED_VERSION = re.compile(r"1\.0\.(0|[1-9][0-9]*)")


def is_supported_version(version: Any) -> bool:
    """True for full AGSi 1.0.x version strings ``"1.0.<patch>"``, e.g. ``"1.0.0"`` or ``"1.0.1"``."""
    return isinstance(version, str) and _SUPPORTED_VERSION.fullmatch(version) is not None


def detect_version(document: dict[str, Any], index: Optional[dict[str, Any]] = None, name: str = "") -> str:
    """Return the AGSi version of a main JSON document, checking it is supported.

    The version is ``agsSchema.version``, which AGSi 1.0.x requires. Versions declared in the
    package ``index.json`` (``schemaVersion``, or ``fileVersion`` as written by pre-1.0
    packages) must also be supported.

    :param document: The parsed main AGSi JSON document.
    :param index: The ``agsiIndex`` object of the package ``index.json``, if any.
    :param name: Source name used in error messages.
    :raise AGSIUnsupportedVersionError: If the document declares no version (pre-1.0 documents
        such as AGSi v0.6 have no ``agsSchema``), or any declared version is not 1.0.x.
    """
    prefix = f"{name}: " if name else ""
    schema = document.get("agsSchema")
    version = schema.get("version") if isinstance(schema, dict) else None
    declared = [("agsSchema.version", version)]
    if index:
        declared += [(f"index.json {key}", index.get(key)) for key in ("schemaVersion", "fileVersion")]
    for source, value in declared:
        if value is not None and not is_supported_version(value):
            raise AGSIUnsupportedVersionError(
                f"{prefix}AGSi version '{value}' ({source}) is not supported; only AGSi {SUPPORTED_VERSIONS} is."
            )
    if version is None:
        raise AGSIUnsupportedVersionError(
            f"{prefix}document has no 'agsSchema.version'. Only AGSi {SUPPORTED_VERSIONS} documents, which must "
            "declare it, are supported (pre-1.0 documents such as AGSi v0.6 are not)."
        )
    return str(version)
