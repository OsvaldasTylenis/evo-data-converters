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

"""Section geometry engine: AGSi 2D section models to Evo geological-sections line geometry.

Places a section on a straight alignment and turns ``agsiGeometryAreaFromLines`` top/bottom
profiles into polygons over shared line chunks. Only straight alignments are supported; curved
alignments and alignments that change direction are rejected with
:class:`AGSIUnsupportedGeometryError`.
"""

from ..reader.errors import AGSIUnsupportedGeometryError
from .alignment import (
    SectionFrame,
    check_lines_only,
    find_model_alignment,
    section_frame_for_model,
    straight_alignment,
)
from .sections import (
    ModelSection,
    SectionGeometry,
    SectionLine,
    SectionUnit,
    SectionVolume,
    build_model_section,
    build_section_geometry,
    profile_to_section,
)

__all__ = [
    "AGSIUnsupportedGeometryError",
    "ModelSection",
    "SectionFrame",
    "SectionGeometry",
    "SectionLine",
    "SectionUnit",
    "SectionVolume",
    "build_model_section",
    "build_section_geometry",
    "check_lines_only",
    "find_model_alignment",
    "profile_to_section",
    "section_frame_for_model",
    "straight_alignment",
]
