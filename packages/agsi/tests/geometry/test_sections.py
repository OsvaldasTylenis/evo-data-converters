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

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evo.data_converters.agsi.geometry import (
    SectionFrame,
    SectionGeometry,
    SectionUnit,
    build_model_section,
    build_section_geometry,
    profile_to_section,
)
from evo.data_converters.agsi.reader import AGSIInvalidDataError, AGSIUnsupportedGeometryError, AgsiPackage


def _line(*points: tuple[float, float]) -> np.ndarray:
    return np.array(points, dtype=np.float64)


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def _rings(geometry: SectionGeometry, key: str) -> list[np.ndarray]:
    return [geometry.ring(v) for v in geometry.volumes if v.key == key]


def _assert_closed(geometry: SectionGeometry) -> None:
    for volume in geometry.volumes:
        ring = geometry.ring(volume)
        np.testing.assert_array_equal(ring[0], ring[-1])


class TestProfileToSection:
    def test_keeps_chainage_and_elevation(self) -> None:
        line = _line((10.0, 1.5), (12.5, -0.25), (20.0, 3.0))
        np.testing.assert_array_equal(profile_to_section(line), line)

    def test_reversed_profile_is_oriented_and_duplicates_removed(self) -> None:
        line = _line((20.0, 3.0), (12.5, -0.25), (12.5, -0.25), (10.0, 1.5))
        np.testing.assert_array_equal(profile_to_section(line), [[10.0, 1.5], [12.5, -0.25], [20.0, 3.0]])

    @pytest.mark.parametrize(
        "line",
        [
            _line((0, 0), (1, 1), (1, 2), (2, 2)),  # vertical step
            _line((0, 0), (2, 1), (1, 2), (3, 2)),  # overhang
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),  # 3D
        ],
    )
    def test_unsupported(self, line: np.ndarray) -> None:
        with pytest.raises(AGSIUnsupportedGeometryError):
            profile_to_section(line)

    @pytest.mark.parametrize("line", [_line((0, 0)), _line((1, 1), (1, 1)), _line((0, 0), (np.inf, 1)), np.zeros(4)])
    def test_invalid(self, line: np.ndarray) -> None:
        with pytest.raises(AGSIInvalidDataError):
            profile_to_section(line)


class TestBuildSectionGeometry:
    def test_shared_contacts(self) -> None:
        t0 = _line((0, 10), (5, 11), (10, 10))
        t1 = _line((0, 5), (5, 4), (10, 5))
        t2 = _line((0, 0), (5, 1), (10, 0))
        units = [SectionUnit("A", t0, t1), SectionUnit("B", t1.copy(), t2), SectionUnit("C", t2.copy())]
        geometry = build_section_geometry(units, base_elevation=-5.0)

        assert [(line.top_of, line.bottom_of, line.generated) for line in geometry.lines] == [
            (["A"], [], False),
            (["B"], ["A"], False),
            (["C"], ["B"], False),
            ([], ["C"], True),
        ]
        # Each line is stored once: 3 profiles + generated base, then 2 edges per unit.
        assert len(geometry.vertices) == 3 * 3 + 2 + 3 * 2 * 2
        contact = geometry.lines[1]
        np.testing.assert_array_equal(geometry.line_vertices(contact), t1)
        volume_a, volume_b = geometry.volumes[0], geometry.volumes[1]
        assert [(c, True) for c in reversed(contact.parts)] == [p for p in volume_a.parts if p[0] in contact.parts]
        assert [(c, False) for c in contact.parts] == [p for p in volume_b.parts if p[0] in contact.parts]

        _assert_closed(geometry)
        ring_a = geometry.ring(volume_a)
        np.testing.assert_array_equal(ring_a, [[0, 10], [5, 11], [10, 10], [10, 5], [5, 4], [0, 5], [0, 10]])
        assert _area(geometry.ring(geometry.volumes[2])) == pytest.approx(5 * 10 + 0.5 * 10)

    def test_pinch_out_splits_unit_and_shares_chunks(self) -> None:
        top = _line((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
        middle = _line((0, -1), (1, 0), (2, 0), (3, -1), (4, -1))
        bottom = _line((0, -2), (4, -2))
        geometry = build_section_geometry([SectionUnit("upper", top, middle), SectionUnit("lower", middle, bottom)])

        rings = _rings(geometry, "upper")
        np.testing.assert_array_equal(rings[0], [[0, 0], [1, 0], [0, -1], [0, 0]])
        np.testing.assert_array_equal(rings[1], [[2, 0], [3, 0], [4, 0], [4, -1], [3, -1], [2, 0]])
        assert sum(_area(r) for r in rings) == pytest.approx(0.5 + 1.5)

        contact = geometry.lines[1]
        assert len(contact.parts) == 3  # split at the pinch-out ends, 1 and 2
        upper_chunks = {c for v in geometry.volumes if v.key == "upper" for c, _ in v.parts}
        (lower,) = (v for v in geometry.volumes if v.key == "lower")
        assert [c for c, r in lower.parts if c in contact.parts] == contact.parts
        assert {contact.parts[0], contact.parts[2]} <= upper_chunks
        assert contact.parts[1] not in upper_chunks
        _assert_closed(geometry)

    def test_pinch_at_single_vertex_splits(self) -> None:
        top = _line((0, 1), (1, 0), (2, 1))
        bottom = _line((0, -1), (1, 0), (2, -1))
        geometry = build_section_geometry([SectionUnit("U", top, bottom)])
        assert [len(r) for r in _rings(geometry, "U")] == [4, 4]

    def test_zero_thickness_unit_has_no_volume(self) -> None:
        line = _line((0, 0), (5, 1))
        geometry = build_section_geometry([SectionUnit("absent", line, line.copy())])
        assert geometry.volumes == []
        assert geometry.lines[0].top_of == ["absent"] and geometry.lines[0].bottom_of == ["absent"]

    def test_top_below_bottom(self) -> None:
        with pytest.raises(AGSIInvalidDataError, match="below bottom at chainage 5"):
            build_section_geometry([SectionUnit("U", _line((0, 1), (5, -1), (9, 1)), _line((0, 0), (5, 0), (9, 0)))])

    def test_different_sampling_exact_crossing_check(self) -> None:
        top = _line((0, 0), (4, 0))
        touching = _line((0, -1), (2, 0), (4, -1))
        geometry = build_section_geometry([SectionUnit("U", top, touching)])
        (ring,) = _rings(geometry, "U")
        np.testing.assert_array_equal(ring, [[0, 0], [4, 0], [4, -1], [2, 0], [0, -1], [0, 0]])

        crossing = _line((0, -1), (2, 0.25), (4, -1))
        with pytest.raises(AGSIInvalidDataError, match="crosses below"):
            build_section_geometry([SectionUnit("U", top, crossing)])
        above_bottom_vertices = _line((0, -1), (1, -0.5), (3, 0.5), (4, 1))
        with pytest.raises(AGSIInvalidDataError, match="crosses below"):
            build_section_geometry([SectionUnit("U", _line((0, 0), (2, -0.25), (4, 1.5)), above_bottom_vertices)])

    def test_different_chainage_range(self) -> None:
        with pytest.raises(AGSIUnsupportedGeometryError, match="same chainage range"):
            build_section_geometry([SectionUnit("U", _line((0, 1), (10, 1)), _line((1, 0), (10, 0)))])

    def test_missing_bottom_and_top(self) -> None:
        top = _line((0, 2), (3, 1), (6, 2))
        with pytest.raises(AGSIInvalidDataError, match="base elevation is required"):
            build_section_geometry([SectionUnit("U", top=top)])
        with pytest.raises(AGSIInvalidDataError, match="crosses below"):
            build_section_geometry([SectionUnit("U", top=top)], base_elevation=1.5)
        (ring,) = _rings(build_section_geometry([SectionUnit("U", top=top)], base_elevation=1.0), "U")
        np.testing.assert_array_equal(ring, [[0, 2], [3, 1], [6, 2], [6, 1], [0, 1], [0, 2]])

        bottom = _line((0, 0), (6, -1))
        with pytest.raises(AGSIInvalidDataError, match="top elevation is required"):
            build_section_geometry([SectionUnit("U", bottom=bottom)])
        geometry = build_section_geometry([SectionUnit("U", bottom=bottom)], top_elevation=4.0)
        (ring,) = _rings(geometry, "U")
        np.testing.assert_array_equal(ring, [[0, 4], [6, 4], [6, -1], [0, 0], [0, 4]])
        assert geometry.lines[0].generated

        with pytest.raises(AGSIInvalidDataError, match="neither"):
            build_section_geometry([SectionUnit("U")])

    def test_empty(self) -> None:
        geometry = build_section_geometry([])
        assert geometry.vertices.shape == (0, 2) and geometry.chunks.shape == (0, 2)


STRAIGHT_NB = "LINESTRING (539591.028 179854.986, 539910.921 180226.890)"


def _straighten(source: Path, target: Path) -> Path:
    """Copy a mini Silvertown package, replacing its curved NB alignment with a straight one."""
    folder = Path(shutil.copytree(source, target))
    (folder / "Alignments" / "NB_Alignment.txt").write_text(STRAIGHT_NB)
    return folder


@pytest.fixture
def straight_package(wkt_dir: Path, tmp_path: Path) -> Path:
    """The mini Silvertown package with its curved NB alignment replaced by a straight one."""
    return _straighten(wkt_dir, tmp_path / "straight")


def _edit_model(folder: Path, edit: Any) -> None:
    path = folder / "Silvertown_mini.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document["agsiModel"][0])
    path.write_text(json.dumps(document), encoding="utf-8")


class TestBuildModelSection:
    def test_straight_package(self, straight_package: Path) -> None:
        package = AgsiPackage.open(straight_package)
        section = build_model_section(package, package.models[0], base_elevation=-50.0)

        assert section.frame.origin_xy == (539591.028, 179854.986)
        assert section.frame.length == math.hypot(539910.921 - 539591.028, 180226.890 - 179854.986)
        assert list(section.elements) == ["Profile Surficial Alignment NB", "Profile Alluvium Alignment NB"]
        geometry = section.geometry
        surficial_base = package.read_geometry("Profiles/NB_Alignment-01-Surficial-Base.txt").single_line()
        alluvium_top = package.read_geometry("Profiles/NB_Alignment-02-Alluvium-Top.txt").single_line()
        np.testing.assert_array_equal(surficial_base, alluvium_top)
        contact = geometry.lines[1]
        assert contact.top_of == ["Profile Alluvium Alignment NB"]
        assert contact.bottom_of == ["Profile Surficial Alignment NB"]
        np.testing.assert_array_equal(geometry.line_vertices(contact), surficial_base)
        assert len(geometry.volumes) == 2
        _assert_closed(geometry)

        world = section.frame.section_to_world(geometry.line_vertices(contact))
        np.testing.assert_allclose(world[0], [539591.028, 179854.986, surficial_base[0, 1]])
        np.testing.assert_array_equal(world[:, 2], surficial_base[:, 1])

    def test_dxf_profiles_match_wkt(self, dxf_dir: Path, wkt_dir: Path, tmp_path: Path) -> None:
        sections = []
        for folder in (wkt_dir, dxf_dir):
            folder = _straighten(folder, tmp_path / folder.name)
            package = AgsiPackage.open(folder)
            sections.append(build_model_section(package, package.models[0], base_elevation=-50.0))
        np.testing.assert_array_equal(sections[0].geometry.vertices, sections[1].geometry.vertices)
        np.testing.assert_array_equal(sections[0].geometry.chunks, sections[1].geometry.chunks)

    def test_base_elevation_required_or_from_boundary(self, straight_package: Path) -> None:
        package = AgsiPackage.open(straight_package)
        with pytest.raises(AGSIInvalidDataError, match="'Profile Alluvium Alignment NB' has no bottom line"):
            build_model_section(package, package.models[0])

        _edit_model(straight_package, lambda m: m.update(agsiModelBoundary={"bottomElevation": -40.0}))
        package = AgsiPackage.open(straight_package)
        section = build_model_section(package, package.models[0])
        assert section.geometry.lines[-1].generated
        np.testing.assert_array_equal(section.geometry.line_vertices(section.geometry.lines[-1])[:, 1], [-40, -40])

    def test_curved_linked_alignment_rejected_even_with_frame(self, wkt_dir: Path) -> None:
        package = AgsiPackage.open(wkt_dir)
        with pytest.raises(AGSIUnsupportedGeometryError, match="not straight"):
            build_model_section(package, package.models[0], base_elevation=-50.0)
        with pytest.raises(AGSIUnsupportedGeometryError, match="agsiModelAlignment 'NB'.*not straight"):
            build_model_section(package, package.models[0], frame=SectionFrame.local(), base_elevation=-50.0)

    def test_explicit_frame_overrides_straight_linked_alignment(self, straight_package: Path) -> None:
        package = AgsiPackage.open(straight_package)
        frame = SectionFrame(origin_xy=(100.0, 200.0), direction_xy=(0.6, 0.8), start_chainage=10.0)
        section = build_model_section(package, package.models[0], frame=frame, base_elevation=-50.0)
        assert section.frame is frame
        assert len(section.geometry.volumes) == 2

    def test_unlinked_model_with_explicit_frame(self, wkt_dir: Path, tmp_path: Path) -> None:
        # The curved alignment is unreferenced once alignmentID is removed, so it is not checked.
        folder = Path(shutil.copytree(wkt_dir, tmp_path / "unlinked"))
        _edit_model(folder, lambda m: m.pop("alignmentID"))
        package = AgsiPackage.open(folder)
        frame = SectionFrame(origin_xy=(100.0, 200.0), direction_xy=(0.6, 0.8))
        section = build_model_section(package, package.models[0], frame=frame, base_elevation=-50.0)
        assert section.frame is frame
        assert len(section.geometry.volumes) == 2

    def test_no_alignment_uses_local_frame(self, straight_package: Path) -> None:
        _edit_model(straight_package, lambda m: m.pop("alignmentID"))
        package = AgsiPackage.open(straight_package)
        section = build_model_section(package, package.models[0], base_elevation=-50.0)
        assert section.frame == SectionFrame.local()

    def test_non_area_elements_skipped(self, straight_package: Path) -> None:
        def add_element(model: dict[str, Any]) -> None:
            model["agsiModelElement"].append(
                {"elementID": "plane", "agsiGeometry": {"elevation": 1.0}, "geometryObject": "agsiGeometryPlane"}
            )

        _edit_model(straight_package, add_element)
        package = AgsiPackage.open(straight_package)
        section = build_model_section(package, package.models[0], base_elevation=-50.0)
        assert "plane" not in section.elements

    def test_3d_profiles_rejected(self, straight_package: Path) -> None:
        profile = straight_package / "Profiles" / "NB_Alignment-02-Alluvium-Top.txt"
        profile.write_text("LINESTRING Z (539591.028 179854.986 1, 539600 179860 2)")
        package = AgsiPackage.open(straight_package)
        with pytest.raises(AGSIUnsupportedGeometryError, match="3D profiles"):
            build_model_section(package, package.models[0], base_elevation=-50.0)
