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

import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from evo_schemas.components import Rotation_V1_1_0

from evo.data_converters.agsi.exporter import AGSIExporterException, AgsiExport, UnsupportedSectionGeometryError
from evo.data_converters.agsi.exporter.geological_sections import (
    assemble_chain,
    assemble_ring,
    linestring_wkt,
    ring_to_profiles,
    section_axes,
)
from evo.data_converters.agsi.geometry import SectionFrame
from evo.data_converters.agsi.reader import AgsiPackage


def lines(top: np.ndarray, bottom: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    return top.tolist(), bottom.tolist()


class TestRingToProfiles:
    RECTANGLE = [(0, 0), (10, 0), (10, 5), (0, 5)]

    @pytest.mark.parametrize(
        "ring", [RECTANGLE, RECTANGLE[::-1], RECTANGLE + RECTANGLE[:1], RECTANGLE[2:] + RECTANGLE[:2]]
    )
    def test_rectangle_any_direction_and_start(self, ring: list[tuple[int, int]]) -> None:
        top, bottom = ring_to_profiles(ring, "unit")
        assert lines(top, bottom) == ([[0, 5], [10, 5]], [[0, 0], [10, 0]])

    def test_repeated_vertices_are_ignored(self) -> None:
        ring = [(0, 0), (0, 0), (5, -1), (10, 0), (10, 0), (10, 5), (0, 5), (0, 0)]
        top, bottom = ring_to_profiles(ring, "unit")
        assert lines(top, bottom) == ([[0, 5], [10, 5]], [[0, 0], [5, -1], [10, 0]])

    def test_pinched_ends(self) -> None:
        # Lens closing to a point at both ends: top and bottom share the end vertices.
        ring = [(0, 1), (5, -1), (10, 2), (5, 4)]
        top, bottom = ring_to_profiles(ring, "lens")
        assert lines(top, bottom) == ([[0, 1], [5, 4], [10, 2]], [[0, 1], [5, -1], [10, 2]])

    def test_collinear_vertices_on_vertical_end_are_dropped(self) -> None:
        ring = [(0, 0), (10, 0), (10, 2), (10, 5), (0, 5), (0, 3)]
        top, bottom = ring_to_profiles(ring, "unit")
        assert lines(top, bottom) == ([[0, 5], [10, 5]], [[0, 0], [10, 0]])

    def test_exact_values_are_kept(self) -> None:
        ring = [(0.1, 0.2 + 0.1), (539910.921, -12.3456789012345), (539910.921, 1e-9), (0.1, 7.0)]
        top, bottom = ring_to_profiles(ring, "unit")
        assert top.tolist() == [[0.1, 7.0], [539910.921, 1e-9]]
        assert bottom.tolist() == [[0.1, 0.2 + 0.1], [539910.921, -12.3456789012345]]

    @pytest.mark.parametrize(
        ("ring", "message"),
        [
            pytest.param([(0, 0), (10, 0), (10, 5), (4, 5), (6, 7), (0, 7)], "strictly increasing", id="overhang"),
            pytest.param([(0, 0), (10, 0), (10, 5), (5, 5), (5, 7), (0, 7)], "strictly increasing", id="inner-step"),
            pytest.param(
                [(0, 0), (10, 0), (10, 2), (2, 2), (2, 4), (10, 4), (10, 6), (0, 6)], "2 separate places", id="c-shape"
            ),
            pytest.param([(0, 0), (10, 0), (10, 5), (0, 5), (0, 6)], "folds back", id="folded-end"),
            pytest.param([(0, 0), (5, 0), (10, 0), (5, 0)], "zero area", id="zero-area"),
            pytest.param([(0, 0), (10, 3), (10, 0), (0, 2)], "below", id="bow-tie"),
            pytest.param([(0, 0), (0, 5), (0, 2)], "zero width", id="vertical"),
            pytest.param([(0, 0), (10, 0), (0, 0)], "fewer than three", id="two-vertices"),
        ],
    )
    def test_ambiguous_or_invalid_polygons_are_rejected(self, ring: list[tuple[int, int]], message: str) -> None:
        with pytest.raises(UnsupportedSectionGeometryError, match=message):
            ring_to_profiles(ring, "unit")

    def test_non_finite_coordinates(self) -> None:
        with pytest.raises(AGSIExporterException, match="non-finite"):
            ring_to_profiles([(0, 0), (10, math.nan), (10, 5)], "unit")


def part(index: int, reversed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(index=index, reversed=reversed)


class TestAssembleRing:
    VERTICES = np.array([(0, 5), (10, 5), (10, 5), (10, 0), (0, 0), (10, 0), (0, 0), (0, 5)], dtype=np.float64)
    # top, right edge, bottom (stored left to right), left edge
    CHUNKS = np.array([(0, 2), (2, 2), (4, 2), (6, 2)])

    def test_reversed_parts(self) -> None:
        ring = assemble_ring(self.VERTICES, self.CHUNKS, [part(0), part(1), part(2, True), part(3)], "v")
        assert ring.tolist() == [[0, 5], [10, 5], [10, 0], [0, 0], [0, 5]]

    @pytest.mark.parametrize(
        ("parts", "message"),
        [
            ([part(0), part(1), part(2), part(3)], "does not start where"),
            ([part(0), part(1), part(2, True)], "closed ring"),
            ([part(9)], "there are 4"),
            ([], "no line parts"),
        ],
    )
    def test_invalid_parts(self, parts: list[SimpleNamespace], message: str) -> None:
        with pytest.raises(AGSIExporterException, match=message):
            assemble_ring(self.VERTICES, self.CHUNKS, parts, "v")

    def test_chunk_outside_vertices(self) -> None:
        with pytest.raises(AGSIExporterException, match="outside the line vertices"):
            assemble_chain(self.VERTICES, np.array([(7, 2)]), [part(0)], "v")


class TestSectionAxes:
    @pytest.mark.parametrize("bearing", [0.0, 40.7, 90.0, 180.0, 233.3, 359.0])
    def test_inverse_of_importer_frame(self, bearing: float) -> None:
        radians = math.radians(bearing)
        frame = SectionFrame((539591.028, 179854.986), (math.sin(radians), math.cos(radians)), start_chainage=12.5)
        section = SimpleNamespace(origin=list(frame.origin), rotation=frame.rotation)
        origin, direction = section_axes(section, "s")
        np.testing.assert_allclose(origin, frame.origin)
        np.testing.assert_allclose(direction, (math.sin(radians), math.cos(radians)), atol=1e-12)

    @pytest.mark.parametrize(
        "rotation",
        [
            Rotation_V1_1_0(dip_azimuth=130.0, dip=80.0, pitch=180.0),
            Rotation_V1_1_0(dip_azimuth=130.0, dip=90.0, pitch=170.0),
            Rotation_V1_1_0(dip_azimuth=0.0, dip=0.0, pitch=0.0),
        ],
    )
    def test_non_vertical_section_is_rejected(self, rotation: Rotation_V1_1_0) -> None:
        section = SimpleNamespace(origin=[0.0, 0.0, 0.0], rotation=rotation)
        with pytest.raises(UnsupportedSectionGeometryError, match="not a vertical plane"):
            section_axes(section, "s")

    def test_invalid_origin(self) -> None:
        rotation = Rotation_V1_1_0(dip_azimuth=0.0, dip=90.0, pitch=180.0)
        section = SimpleNamespace(origin=[0.0, math.inf, 0.0], rotation=rotation)
        with pytest.raises(AGSIExporterException, match="three finite"):
            section_axes(section, "s")


def test_linestring_wkt_is_round_trip_safe() -> None:
    values = [0.1 + 0.2, 539910.921, 1e-9, -0.0, 123456789.12345679, -7.25, 1e16]
    text = linestring_wkt(np.array([values, values]).T)
    assert text.startswith("LINESTRING (")
    coordinates = text.removeprefix("LINESTRING ")
    assert "e" not in coordinates.lower() and "-0 " not in coordinates
    parsed = [float(token.strip(",()")) for token in coordinates.split()]
    assert parsed[::2] == [value + 0.0 for value in values]


def test_write_package(tmp_path: Path) -> None:
    document = {
        "agsSchema": {"name": "AGSi", "version": "1.0.1"},
        "agsFile": {"title": "Ünïcode title", "producedBy": "tests"},
        "agsiModel": [],
    }
    export = AgsiExport(title="Ünïcode title", main_file="main.json", document=document, files={"a/b.wkt": b"x"})
    path = tmp_path / "out.agsi"
    export.write(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["index.json", "main.json", "a/b.wkt"]
    package = AgsiPackage.open(path)
    assert package.schema_version == "1.0.1"
    assert package.read_bytes("a/b.wkt") == b"x"
