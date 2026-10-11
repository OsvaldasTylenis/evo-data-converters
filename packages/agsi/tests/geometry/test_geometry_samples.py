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

"""Geometry-engine integration tests against the AGSi 1.0.1 Silvertown packages (see ``test_agsi_samples``).

The Silvertown NB/SB alignments are curved, so the 2D sections are rejected. The optional
``Silvertown2D_wkt_straight_synthetic.agsi`` package (the same model on a straight alignment)
is built end to end when present.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from evo.data_converters.agsi.geometry import build_model_section
from evo.data_converters.agsi.reader import AGSIUnsupportedGeometryError, AgsiPackage

SAMPLE_DIR = os.environ.get("AGSI_SAMPLE_DIR")
CURVED_2D = ["Silvertown2D_wkt_1", "Silvertown2D_wkt_3", "Silvertown2D_dxf_1", "Silvertown2D_dxf_3"]
STRAIGHT_2D = "Silvertown2D_wkt_straight_synthetic"

pytestmark = pytest.mark.skipif(not SAMPLE_DIR, reason="Set AGSI_SAMPLE_DIR to run the AGSi sample tests")


def _open(name: str) -> AgsiPackage:
    path = Path(str(SAMPLE_DIR)) / f"{name}.agsi"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return AgsiPackage.open(path)


@pytest.mark.parametrize("name", CURVED_2D)
def test_curved_alignments_rejected(name: str) -> None:
    package = _open(name)
    section_model = next(m for m in package.models if m.alignment_id)
    with pytest.raises(AGSIUnsupportedGeometryError, match="not straight"):
        build_model_section(package, section_model, base_elevation=-100.0)


def test_straight_synthetic_section() -> None:
    package = _open(STRAIGHT_2D)
    section_model = next(m for m in package.models if m.alignment_id)
    section = build_model_section(package, section_model, base_elevation=-100.0)

    start, end = (539591.028, 179854.986), (539910.921, 180226.890)
    assert section.frame.origin_xy == start
    assert section.frame.length == math.hypot(end[0] - start[0], end[1] - start[1])

    geometry = section.geometry
    assert len(section.elements) == 11
    assert sum(1 for line in geometry.lines if line.top_of and line.bottom_of) == 10
    assert sum(1 for line in geometry.lines if line.generated) == 1
    assert {v.key for v in geometry.volumes} == set(section.elements)
    for volume in geometry.volumes:
        ring = geometry.ring(volume)
        np.testing.assert_array_equal(ring[0], ring[-1])
