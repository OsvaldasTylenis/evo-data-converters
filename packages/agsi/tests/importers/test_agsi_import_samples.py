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

"""Offline importer tests against the AGSi 1.0.1 Silvertown packages (see ``test_agsi_samples``).

Skipped unless ``AGSI_SAMPLE_DIR`` points at the converted sample folder. Objects are built with a
local data client only; nothing is published. EPSG:27700 is supplied by the caller: the samples do
not state their CRS (it is inferred in their remarks).
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from evo_schemas.components import Crs_V1_0_1_EpsgCode
from evo_schemas.objects.geological_sections import GeologicalSections_V1_2_0

from evo.data_converters.agsi.importer.utils import get_geoscience_object_from_agsi
from evo.data_converters.agsi.reader import AGSIUnsupportedGeometryError
from evo.data_converters.common import crs_from_epsg_code

SAMPLE_DIR = os.environ.get("AGSI_SAMPLE_DIR")
CURVED_2D = ["Silvertown2D_wkt_1", "Silvertown2D_wkt_3", "Silvertown2D_dxf_1", "Silvertown2D_dxf_3"]
SAMPLES_3D = ["Silvertown3D_wkt", "Silvertown3D_dxf", "Silvertown3D_stl"]
STRAIGHT_2D = "Silvertown2D_wkt_straight_synthetic"

pytestmark = [
    pytest.mark.skipif(not SAMPLE_DIR, reason="Set AGSI_SAMPLE_DIR to run the AGSi sample tests"),
    pytest.mark.usefixtures("block_network"),
]


def _path(name: str) -> str:
    path = Path(str(SAMPLE_DIR)) / f"{name}.agsi"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return str(path)


def test_straight_synthetic_sample(local_data_client: MagicMock) -> None:
    obj = get_geoscience_object_from_agsi(
        local_data_client, _path(STRAIGHT_2D), crs_from_epsg_code(27700), base_elevation=-100.0
    )
    assert isinstance(obj, GeologicalSections_V1_2_0)
    assert obj.coordinate_reference_system == Crs_V1_0_1_EpsgCode(epsg_code=27700)
    assert len(obj.sections) == 1
    assert len(obj.materials or []) == 11 and len(obj.layer_order or []) == 11
    # Pinched-out units are split into several volumes.
    assert {volume.material_key for volume in obj.volumes} == set(obj.layer_order or [])
    assert len(obj.volumes) >= 11
    # 11 top lines and one generated base for the bottom unit.
    assert len(obj.surfaces) == 12
    assert sum(surface.feature == "BoundarySurface" for surface in obj.surfaces) == 1
    box = obj.bounding_box
    assert (box.min_x, box.min_y, box.max_x, box.max_y) == (539591.028, 179854.986, 539910.921, 180226.89)
    assert box.min_z == -100.0
    assert GeologicalSections_V1_2_0.from_dict(obj.as_dict()) == obj


@pytest.mark.parametrize("name", CURVED_2D)
def test_curved_samples_rejected(local_data_client: MagicMock, name: str) -> None:
    with pytest.raises(AGSIUnsupportedGeometryError, match="not straight"):
        get_geoscience_object_from_agsi(
            local_data_client, _path(name), crs_from_epsg_code(27700), base_elevation=-100.0
        )
    assert not local_data_client.save_table.called


@pytest.mark.parametrize("name", SAMPLES_3D)
def test_3d_samples_rejected(local_data_client: MagicMock, name: str) -> None:
    # The 3D models hold AreaFromLines elements with 3D (x, y, z) profiles, which are not sections.
    with pytest.raises(AGSIUnsupportedGeometryError, match="3D profiles are not supported"):
        get_geoscience_object_from_agsi(local_data_client, _path(name), crs_from_epsg_code(27700))
    assert not local_data_client.save_table.called
