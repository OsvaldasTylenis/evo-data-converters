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

"""Integration tests against the Silvertown example packages converted to AGSi 1.0.1.

The packages (up to ~80 MB) are not committed. The examples published on the AGSi guidance pages
use the pre-1.0 AGSi v0.6 format, which this reader does not support; point ``AGSI_SAMPLE_DIR`` at
a folder holding their AGSi 1.0.1 conversions (same ``Silvertown2D_*`` / ``Silvertown3D_*`` file
names) to run these tests. In the converted 2D packages the section model references, by
``alignmentID``, the ``agsiModelAlignment`` of a separate plan model in British National Grid.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from evo.data_converters.agsi.reader import AgsiGeometryFromFile, AgsiModel, AgsiPackage

SAMPLE_DIR = os.environ.get("AGSI_SAMPLE_DIR")
SAMPLES_2D = ["Silvertown2D_wkt_1", "Silvertown2D_wkt_3", "Silvertown2D_dxf_1", "Silvertown2D_dxf_3"]
SAMPLES_3D = ["Silvertown3D_wkt", "Silvertown3D_dxf", "Silvertown3D_stl"]
# End chainage of each section alignment (metres along the plan alignment).
ALIGNMENT_LENGTHS = {"NB": 1899.6905741668209, "SB": 1887.631157393874}
PROFILE_VERTICES = 401

pytestmark = pytest.mark.skipif(not SAMPLE_DIR, reason="Set AGSI_SAMPLE_DIR to run the AGSi sample tests")

AREA = "agsiGeometryAreaFromLines"
VOLUME = "agsiGeometryVolFromSurfaces"


def _open(name: str) -> AgsiPackage:
    path = Path(str(SAMPLE_DIR)) / f"{name}.agsi"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    package = AgsiPackage.open(path)
    assert package.schema_version == "1.0.1"
    assert package.warnings == []
    return package


def _file_refs(model: AgsiModel, geometry_object: str) -> list[AgsiGeometryFromFile]:
    return [
        geometry
        for element in model.elements_by_geometry(geometry_object)
        for role, geometry in element.geometry_files()
        if role in ("top", "bottom")
    ]


@pytest.mark.parametrize("name", SAMPLES_2D)
def test_2d_sample_package(name: str) -> None:
    package = _open(name)
    document = package.document
    section, plan = package.models
    alignment_id = "NB" if name.endswith("_1") else "SB"

    assert section.alignment_id == alignment_id
    assert len(section.agsi_model_element) == 11
    section_system = document.coordinate_system(section)
    assert section_system is not None and section_system.system_id == f"{alignment_id}-Section-XZ"

    assert plan.agsi_model_element == []
    plan_system = document.coordinate_system(plan)
    assert plan_system is not None and plan_system.global_xy_system == "EPSG:27700"
    (alignment,) = plan.agsi_model_alignment
    assert alignment.alignment_id == alignment_id and alignment.start_chainage == 0
    assert alignment.agsi_geometry is not None
    path = package.read_geometry(alignment.agsi_geometry).single_line()
    assert path.shape == (PROFILE_VERTICES, 2)
    station = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    assert station[-1] == pytest.approx(ALIGNMENT_LENGTHS[alignment_id], abs=1e-6)

    profiles = _file_refs(section, AREA)
    assert len(profiles) == 21
    for profile in profiles:
        line = package.read_geometry(profile).single_line()
        assert line.shape == (PROFILE_VERTICES, 2)
        # Profile chainage is the horizontal station along the plan alignment.
        np.testing.assert_allclose(line[:, 0], station, rtol=0, atol=1e-6)


@pytest.mark.parametrize("suffix", ["1", "3"])
def test_2d_wkt_and_dxf_profiles_match(suffix: str) -> None:
    wkt, dxf = _open(f"Silvertown2D_wkt_{suffix}"), _open(f"Silvertown2D_dxf_{suffix}")
    wkt_refs, dxf_refs = _file_refs(wkt.models[0], AREA), _file_refs(dxf.models[0], AREA)
    assert len(wkt_refs) == len(dxf_refs)
    for wkt_ref, dxf_ref in zip(wkt_refs, dxf_refs):
        np.testing.assert_allclose(
            wkt.read_geometry(wkt_ref).single_line(), dxf.read_geometry(dxf_ref).single_line(), rtol=0, atol=1e-9
        )


@pytest.mark.parametrize("name", SAMPLES_3D)
def test_3d_sample_package(name: str) -> None:
    package = _open(name)
    (model,) = package.models
    system = package.document.coordinate_system(model)
    assert system is not None and system.system_id == "BNG-XYZ"
    assert len(model.agsi_model_element) == 33
    assert len(model.elements_by_geometry(VOLUME)) == 11
    assert len(model.elements_by_geometry(AREA)) == 22

    surfaces = _file_refs(model, VOLUME)
    assert len(surfaces) == 21
    mesh = package.read_geometry(surfaces[0]).to_triangle_mesh()
    assert len(mesh.triangles) > 0 and mesh.vertices.shape[1] == 3

    boundary = model.agsi_model_boundary
    assert boundary is not None and boundary.agsi_geometry_boundary_xy is not None
    assert package.read_geometry(boundary.agsi_geometry_boundary_xy).dimension == 3

    profiles = _file_refs(model, AREA)
    assert len(profiles) == 42
    for profile in profiles:
        assert package.read_geometry(profile).single_line().shape == (PROFILE_VERTICES, 3)