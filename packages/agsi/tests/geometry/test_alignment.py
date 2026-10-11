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

import io
import math
import random
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evo.data_converters.agsi.geometry import (
    SectionFrame,
    find_model_alignment,
    section_frame_for_model,
    straight_alignment,
)
from evo.data_converters.agsi.reader import (
    AGSIInvalidDataError,
    AGSIUnsupportedGeometryError,
    AgsiDocument,
    AgsiPackage,
    parse_dxf,
    parse_wkt,
)
from evo.data_converters.common.utils import convert_rotation


def _dxf(build: Any) -> bytes:
    doc = ezdxf.new()
    build(doc.modelspace())
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def _evo_matrix(frame: SectionFrame) -> np.ndarray:
    """Local-to-world rotation of an Evo rotation (intrinsic ZXZ, clockwise angles)."""
    r = frame.rotation
    matrix: np.ndarray = Rotation.from_euler("ZXZ", [-r.dip_azimuth, -r.dip, -r.pitch], degrees=True).as_matrix()
    return matrix


class TestSectionFrame:
    @pytest.mark.parametrize("bearing", [0.0, 30.0, 90.0, 135.0, 180.0, 225.0, 270.0, 359.0])
    def test_rotation_maps_u_along_alignment_and_v_up(self, bearing: float) -> None:
        b = math.radians(bearing)
        frame = SectionFrame(origin_xy=(0.0, 0.0), direction_xy=(math.sin(b), math.cos(b)))
        assert frame.bearing == pytest.approx(bearing)
        matrix = _evo_matrix(frame)
        np.testing.assert_allclose(matrix[:, 0], [math.sin(b), math.cos(b), 0.0], atol=1e-12)
        np.testing.assert_allclose(matrix[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
        # Same angles as the shared converter helper for the matrix with columns (u, v, u x v).
        u, v = matrix[:, 0], matrix[:, 1]
        expected = convert_rotation(Rotation.from_matrix(np.column_stack([u, v, np.cross(u, v)])))
        assert frame.rotation.dip_azimuth == pytest.approx(expected.dip_azimuth % 360.0, abs=1e-9)
        assert frame.rotation.dip == pytest.approx(expected.dip)
        assert frame.rotation.pitch == pytest.approx(expected.pitch)

    def test_section_world_round_trip(self) -> None:
        frame = SectionFrame(origin_xy=(1000.0, 2000.0), direction_xy=(0.6, 0.8), start_chainage=50.0, length=10.0)
        world = frame.section_to_world([[50.0, -3.0], [60.0, 4.0], [0.0, 1.0]])
        np.testing.assert_allclose(world, [[1000, 2000, -3], [1006, 2008, 4], [970, 1960, 1]])
        assert frame.origin == pytest.approx((970.0, 1960.0, 0.0))
        assert frame.end_chainage == 60.0
        np.testing.assert_allclose(frame.chainage_to_xy([55.0]), [[1003.0, 2004.0]])
        uv, offset = frame.world_to_section(np.vstack([world, [1000.8, 1999.4, 2.0]]))
        np.testing.assert_allclose(uv, [[50, -3], [60, 4], [0, 1], [50, 2]], atol=1e-9)
        np.testing.assert_allclose(offset, [0, 0, 0, 1.0], atol=1e-9)

    def test_local_frame(self) -> None:
        frame = SectionFrame.local()
        np.testing.assert_array_equal(frame.section_to_world([[12.5, -4.0]]), [[12.5, 0.0, -4.0]])
        assert frame.origin == (0.0, 0.0, 0.0)
        assert frame.bearing == 90.0
        assert frame.length is None and frame.end_chainage is None


class TestStraightAlignment:
    def test_two_points(self) -> None:
        frame = straight_alignment(parse_wkt(b"LINESTRING (539591.028 179854.986, 539910.921 180226.890)"), 5.0)
        assert frame.origin_xy == (539591.028, 179854.986)
        assert frame.length == math.hypot(539910.921 - 539591.028, 180226.890 - 179854.986)
        assert frame.start_chainage == 5.0
        np.testing.assert_allclose(frame.chainage_to_xy([5.0 + frame.length]), [[539910.921, 180226.890]])

    def test_direction_follows_vertex_order(self) -> None:
        forward = straight_alignment([[[0, 0], [3, 4]]])
        backward = straight_alignment([[[3, 4], [0, 0]]])
        assert forward.direction_xy == (0.6, 0.8)
        assert backward.direction_xy == (-0.6, -0.8)
        assert backward.origin_xy == (3.0, 4.0)

    def test_collinear_vertices_and_z_ignored(self) -> None:
        frame = straight_alignment(parse_wkt(b"LINESTRING Z (0 0 5, 1 1 6, 1 1 6, 2.5 2.5 1, 4 4 0)"))
        assert frame.length == pytest.approx(math.sqrt(32))
        assert frame.bearing == pytest.approx(45.0)

    def test_collinearity_uses_written_decimal_values(self) -> None:
        # (1, 0.1) and (3, 0.3) are collinear with the origin as written, but not as binary floats.
        assert 3 * 0.1 - 1 * 0.3 != 0
        frame = straight_alignment(parse_wkt(b"LINESTRING (0 0, 1 0.1, 3 0.3)"))
        assert frame.length == math.hypot(3, 0.3)

    def test_dxf_line_chain_in_any_order_and_orientation(self) -> None:
        points = [(10.0 + 2 * i, 20.0 + i) for i in range(6)]
        segments = [(points[i], points[i + 1]) for i in range(5)]
        random.Random(1).shuffle(segments)

        def build(msp: Any) -> None:
            for i, (a, b) in enumerate(segments):
                msp.add_line(*((b, a) if i % 2 else (a, b)))

        frame = straight_alignment(parse_dxf(_dxf(build)))
        assert frame.length == pytest.approx(math.hypot(10, 5))
        # Without a vertex order, the direction follows the first LINE entity in the file.
        assert frame.origin_xy in (points[0], points[-1])

    def test_dxf_lwpolyline(self) -> None:
        frame = straight_alignment(parse_dxf(_dxf(lambda msp: msp.add_lwpolyline([(0, 0), (5, 0), (8, 0)]))))
        assert frame.length == 8.0 and frame.bearing == 90.0

    @pytest.mark.parametrize(
        "wkt, message",
        [
            ("LINESTRING (0 0, 1 0, 2 1)", "not straight"),  # kink
            ("LINESTRING (0 0, 2 0, 1 0, 3 0)", "reverses direction"),  # fold-back
            ("LINESTRING (0 0, 2 0, 0 0)", "closed"),
            ("MULTILINESTRING ((0 0, 1 0), (2 0, 3 0))", "separate or branched"),  # disconnected
            ("MULTILINESTRING ((0 0, 1 0), (1 0, 2 0), (1 0, 1 1))", "separate or branched"),  # branched
        ],
    )
    def test_rejects_non_straight(self, wkt: str, message: str) -> None:
        with pytest.raises(AGSIUnsupportedGeometryError, match=message):
            straight_alignment(parse_wkt(wkt.encode()))

    def test_rejects_sampled_arc(self) -> None:
        angles = np.linspace(0, math.pi / 8, 9)
        arc = np.column_stack([1000 * np.sin(angles), 1000 * (1 - np.cos(angles))])
        with pytest.raises(AGSIUnsupportedGeometryError, match="not straight"):
            straight_alignment([arc])

    def test_rejects_dxf_arc_and_bulge(self) -> None:
        with pytest.raises(AGSIUnsupportedGeometryError, match="curved"):
            straight_alignment(parse_dxf(_dxf(lambda msp: msp.add_lwpolyline([(0, 0, 0.4), (5, 0, 0)], format="xyb"))))

        def line_and_arc(msp: Any) -> None:
            msp.add_line((0, 0), (5, 0))
            msp.add_arc((5, 5), 5, 270, 360)

        with pytest.raises(AGSIUnsupportedGeometryError, match="ARC"):
            straight_alignment(parse_dxf(_dxf(line_and_arc)))

    @pytest.mark.parametrize(
        "lines",
        [[], [np.empty((0, 2))], [[[1.0, 1.0], [1.0, 1.0]]], [[[0.0, 0.0], [np.nan, 1.0]]]],
    )
    def test_rejects_invalid(self, lines: list[Any]) -> None:
        with pytest.raises(AGSIInvalidDataError):
            straight_alignment(lines)

    def test_rejects_non_line_geometry(self) -> None:
        with pytest.raises(AGSIInvalidDataError, match="only lines"):
            straight_alignment(parse_wkt(b"POLYGON ((0 0, 1 0, 1 1, 0 0))"))


def _document(alignment_ids: list[str], reference: str | None = "NB") -> AgsiDocument:
    return AgsiDocument.model_validate(
        {
            "agsiModel": [
                {"modelID": "section", "alignmentID": reference},
                {
                    "modelID": "plan",
                    "agsiModelAlignment": [
                        {"alignmentID": a, "agsiGeometry": {"fileURI": "a.txt"}} for a in alignment_ids
                    ],
                },
            ]
        }
    )


class TestModelAlignment:
    def test_resolves_alignment_in_another_model(self) -> None:
        document = _document(["SB", "NB"])
        alignment = find_model_alignment(document, document.agsi_model[0])
        assert alignment is not None and alignment.alignment_id == "NB"

    def test_no_alignment_id(self) -> None:
        document = _document(["NB"], reference=None)
        assert find_model_alignment(document, document.agsi_model[0]) is None

    @pytest.mark.parametrize("ids, message", [(["SB"], "does not match"), (["NB", "NB"], "matches 2")])
    def test_missing_or_ambiguous(self, ids: list[str], message: str) -> None:
        document = _document(ids)
        with pytest.raises(AGSIInvalidDataError, match=message):
            find_model_alignment(document, document.agsi_model[0])

    def test_curved_silvertown_alignment_is_rejected(self, wkt_dir: Path) -> None:
        package = AgsiPackage.open(wkt_dir)
        with pytest.raises(AGSIUnsupportedGeometryError, match="agsiModelAlignment 'NB': Alignment is not straight"):
            section_frame_for_model(package, package.models[0])

    def test_plan_model_without_alignment_id(self, wkt_dir: Path) -> None:
        package = AgsiPackage.open(wkt_dir)
        assert section_frame_for_model(package, package.models[1]) is None
