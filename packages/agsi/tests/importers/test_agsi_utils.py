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

"""Offline tests of the AGSi -> Evo geological-sections object builder (no Evo services are used)."""

import copy
import logging
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pyarrow.parquet as pq
import pytest
from evo_schemas.components import ContinuousAttribute_V1_1_0, Crs_V1_0_1_EpsgCode, StringAttribute_V1_1_0
from evo_schemas.objects.geological_sections import GeologicalSections_V1_2_0

from conftest import DXF_PACKAGE_DIR, edit_document, straighten
from evo.data_converters.agsi.geometry import build_model_section
from evo.data_converters.agsi.importer.geological_sections import evo_colour
from evo.data_converters.agsi.importer.utils import get_geoscience_object_from_agsi
from evo.data_converters.agsi.reader import (
    AGSIDataFileIOError,
    AGSIInvalidDataError,
    AGSIUnsupportedGeometryError,
    AGSIUnsupportedVersionError,
    AgsiPackage,
)
from evo.data_converters.common import crs_from_epsg_code

pytestmark = pytest.mark.usefixtures("block_network")

START = (539591.028, 179854.986)
END = (539910.921, 180226.890)
SURFICIAL = "Profile Surficial Alignment NB"
ALLUVIUM = "Profile Alluvium Alignment NB"


def _table(client: MagicMock, info: Any) -> np.ndarray:
    """Read back a table saved by the local data client as an (N, columns) array."""
    table = pq.read_table(client.cache_location / str(info.data))
    return np.column_stack([column.to_numpy(zero_copy_only=False) for column in table.columns])


def _values(client: MagicMock, attribute: Any) -> list[Any]:
    values: list[Any] = pq.read_table(client.cache_location / str(attribute.values.data)).column(0).to_pylist()
    return values


def _build(client: MagicMock, folder: Path, **kwargs: Any) -> GeologicalSections_V1_2_0:
    kwargs.setdefault("base_elevation", -50.0)
    result = get_geoscience_object_from_agsi(client, str(folder), crs_from_epsg_code(27700), **kwargs)
    assert isinstance(result, GeologicalSections_V1_2_0)
    return result


def _section_model(document: dict[str, Any]) -> dict[str, Any]:
    model: dict[str, Any] = document["agsiModel"][0]
    return model


class TestStraightSection:
    def test_object_layout(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        obj = _build(local_data_client, straight_wkt_dir, tags={"Project": "Silvertown"})

        assert obj.name == "Silvertown mini NB section"
        assert obj.uuid is None
        assert obj.coordinate_reference_system == Crs_V1_0_1_EpsgCode(epsg_code=27700)
        assert obj.tags == {
            "Source": "straight_wkt (via Evo Data Converters)",
            "Stage": "Experimental",
            "InputType": "AGSI",
            "Project": "Silvertown",
        }
        assert obj.extensions == {
            "agsi": {
                "schemaVersion": "1.0.1",
                "modelID": "Silvertown-mini-NB-Section",
                "modelName": "Silvertown mini NB section",
                "coordSystemID": "NB-Section-XZ",
                "alignmentID": "NB",
                "alignmentName": "NB alignment",
                "startChainage": 0.0,
                "alignmentLength": math.hypot(END[0] - START[0], END[1] - START[1]),
            }
        }
        assert [folder.name for folder in obj.folders] == ["NB alignment"]
        assert [item.section_index for item in obj.folders[0].items] == [0]

        (section,) = obj.sections
        package = AgsiPackage.open(straight_wkt_dir)
        frame = build_model_section(package, package.models[0], base_elevation=-50.0).frame
        assert section.name == obj.name
        assert section.origin == [START[0], START[1], 0.0]
        assert section.rotation == frame.rotation
        assert (section.volumes.offset, section.volumes.count) == (0, 2)
        assert section.surfaces is not None and (section.surfaces.offset, section.surfaces.count) == (0, 3)

        assert [(v.name, v.feature, v.material_key, v.layer) for v in obj.volumes] == [
            (SURFICIAL, "OutputVolume", SURFICIAL, SURFICIAL),
            (ALLUVIUM, "OutputVolume", ALLUVIUM, ALLUVIUM),
        ]
        assert obj.layer_order == [SURFICIAL, ALLUVIUM]
        assert [m.key for m in obj.materials or []] == [SURFICIAL, ALLUVIUM]
        assert [(s.name, s.feature, s.description) for s in obj.surfaces] == [
            (f"Top of {SURFICIAL}", "ContactSurface", "Top of Surficial"),
            (f"Base of {SURFICIAL} / Top of {ALLUVIUM}", "ContactSurface", "Base of Surficial; Top of Alluvium"),
            (
                f"Base of {ALLUVIUM}",
                "BoundarySurface",
                "Horizontal line at elevation -50.0 generated for units without a bottom profile.",
            ),
        ]
        # The schema-validated object survives a JSON round trip.
        assert GeologicalSections_V1_2_0.from_dict(obj.as_dict()) == obj

    def test_line_geometry_matches_profiles(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        obj = _build(local_data_client, straight_wkt_dir)
        package = AgsiPackage.open(straight_wkt_dir)
        geometry = build_model_section(package, package.models[0], base_elevation=-50.0).geometry

        vertices = _table(local_data_client, obj.line_geometry.vertices)
        chunks = _table(local_data_client, obj.line_geometry.chunks)
        np.testing.assert_array_equal(vertices, geometry.vertices)
        np.testing.assert_array_equal(chunks, geometry.chunks)
        assert obj.line_geometry.vertices.data_type == "float64" and obj.line_geometry.vertices.width == 2
        assert obj.line_geometry.chunks.data_type == "uint64" and obj.line_geometry.chunks.width == 2

        # Volume and surface parts index chunks; the contact line is one shared chunk.
        def line(parts: list[Any]) -> np.ndarray:
            pieces = [vertices[o : o + c][:: -1 if p.reversed else 1] for p in parts for o, c in [chunks[p.index]]]
            return np.concatenate([pieces[0], *(piece[1:] for piece in pieces[1:])])

        top = package.read_geometry("Profiles/NB_Alignment-01-Surficial-Top.txt").single_line()
        contact = package.read_geometry("Profiles/NB_Alignment-01-Surficial-Base.txt").single_line()
        np.testing.assert_array_equal(line(obj.surfaces[0].parts), top)
        np.testing.assert_array_equal(line(obj.surfaces[1].parts), contact)
        surficial_parts = {(p.index, p.reversed) for p in obj.volumes[0].parts}
        alluvium_parts = {(p.index, p.reversed) for p in obj.volumes[1].parts}
        (shared,) = obj.surfaces[1].parts
        assert (shared.index, True) in surficial_parts and (shared.index, False) in alluvium_parts
        for volume in obj.volumes:
            ring = line(volume.parts)
            np.testing.assert_array_equal(ring[0], ring[-1])

        # Bounding box of the section placed in world coordinates.
        world = np.column_stack(
            [
                START[0] + vertices[:, 0] * (END[0] - START[0]) / obj.extensions["agsi"]["alignmentLength"],
                START[1] + vertices[:, 0] * (END[1] - START[1]) / obj.extensions["agsi"]["alignmentLength"],
                vertices[:, 1],
            ]
        )
        box = obj.bounding_box
        np.testing.assert_allclose([box.min_x, box.min_y, box.min_z], world.min(axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose([box.max_x, box.max_y, box.max_z], world.max(axis=0), rtol=0, atol=1e-6)
        assert (box.min_x, box.min_y, box.min_z) == (START[0], START[1], -50.0)

    def test_saves_tables_locally(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        obj = _build(local_data_client, straight_wkt_dir)
        # Vertices, chunks and one table per volume attribute; no upload happens here.
        assert local_data_client.save_table.call_count == 2 + len(obj.volume_attributes or [])
        assert not local_data_client.upload_referenced_data.called
        saved = {path.name for path in local_data_client.cache_location.iterdir()}
        assert {str(obj.line_geometry.vertices.data), str(obj.line_geometry.chunks.data)} <= saved

    def test_dxf_profiles(self, local_data_client: MagicMock, straight_wkt_dir: Path, tmp_path: Path) -> None:
        dxf = straighten(DXF_PACKAGE_DIR, tmp_path / "straight_dxf")
        wkt_obj, dxf_obj = _build(local_data_client, straight_wkt_dir), _build(local_data_client, dxf)
        assert dxf_obj.line_geometry.vertices.data == wkt_obj.line_geometry.vertices.data
        assert dxf_obj.line_geometry.chunks.data == wkt_obj.line_geometry.chunks.data


class TestElementMetadata:
    @staticmethod
    def _enrich(document: dict[str, Any]) -> None:
        surficial, alluvium = _section_model(document)["agsiModelElement"]
        surficial.update(
            elementName="Surficial",
            colourRGB="#FF8000",
            agsiDataParameterValue=[
                {"codeID": "UnitWeight", "valueNumeric": 19.5},
                {"codeID": "Description", "valueText": "Made ground"},
                {"codeID": "UnitWeight", "caseID": "Upper", "valueNumeric": 20.0},
                {"codeID": "Mixed", "valueNumeric": 1.5},
            ],
        )
        alluvium.update(
            elementName="Alluvium",
            agsiDataParameterValue=[
                {"codeID": "UnitWeight", "valueNumeric": 17.0},
                {"codeID": "Mixed", "valueText": "soft"},
                {"codeID": "Profiled", "valueProfileIndVarCodeID": "Depth", "valueProfile": [[0.0, 1.0]]},
            ],
        )

    def test_names_materials_and_attributes(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        edit_document(straight_wkt_dir, self._enrich)
        with caplog.at_level(logging.WARNING):
            obj = _build(local_data_client, straight_wkt_dir)
        assert "valueProfile of 'Profiled' is not converted" in caplog.text

        assert [v.name for v in obj.volumes] == ["Surficial", "Alluvium"]
        assert [v.material_key for v in obj.volumes] == [SURFICIAL, ALLUVIUM]
        assert [(m.name, m.color) for m in obj.materials or []] == [
            ("Surficial", 0xFF0080FF),  # #FF8000 as alpha, blue, green, red
            ("Alluvium", evo_colour(None, 1)),
        ]
        assert obj.surfaces[1].name == "Base of Surficial / Top of Alluvium"

        attributes = {a.name: a for a in obj.volume_attributes or []}
        assert list(attributes) == [
            "elementID",
            "elementType",
            "UnitWeight",
            "Description",
            "UnitWeight (Upper)",
            "Mixed",
        ]
        assert _values(local_data_client, attributes["elementID"]) == [SURFICIAL, ALLUVIUM]
        assert _values(local_data_client, attributes["elementType"]) == ["Geological unit", "Geological unit"]
        assert isinstance(attributes["UnitWeight"], ContinuousAttribute_V1_1_0)
        assert _values(local_data_client, attributes["UnitWeight"]) == [19.5, 17.0]
        upper = _values(local_data_client, attributes["UnitWeight (Upper)"])
        assert upper[0] == 20.0 and math.isnan(upper[1])
        assert isinstance(attributes["Description"], StringAttribute_V1_1_0)
        assert _values(local_data_client, attributes["Description"]) == ["Made ground", None]
        assert isinstance(attributes["Mixed"], StringAttribute_V1_1_0)
        assert _values(local_data_client, attributes["Mixed"]) == ["1.5", "soft"]
        for attribute in attributes.values():
            assert attribute.values.length == len(obj.volumes)

    @pytest.mark.parametrize(
        "colour, expected",
        [("#FF0000", 0xFF0000FF), ("00ff00", 0xFF00FF00), ("#0000FF", 0xFFFF0000), (" #102030 ", 0xFF302010)],
    )
    def test_evo_colour(self, colour: str, expected: int) -> None:
        assert evo_colour(colour, 0) == expected

    def test_invalid_colour_uses_palette(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            assert evo_colour("red", 2) == evo_colour(None, 2)
        assert "colourRGB 'red'" in caplog.text


class TestElevations:
    def test_base_elevation_required(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        with pytest.raises(AGSIInvalidDataError, match=f"'{ALLUVIUM}' has no bottom line"):
            _build(local_data_client, straight_wkt_dir, base_elevation=None)

    def test_base_from_model_boundary_or_argument(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        edit_document(
            straight_wkt_dir, lambda d: _section_model(d).update(agsiModelBoundary={"bottomElevation": -40.0})
        )
        for base, expected in ((None, -40.0), (-60.0, -60.0)):
            obj = _build(local_data_client, straight_wkt_dir, base_elevation=base)
            base_line = obj.surfaces[-1]
            assert base_line.feature == "BoundarySurface"
            vertices = _table(local_data_client, obj.line_geometry.vertices)
            chunks = _table(local_data_client, obj.line_geometry.chunks)
            offset, count = chunks[base_line.parts[0].index]
            np.testing.assert_array_equal(vertices[offset : offset + count, 1], [expected, expected])
            assert obj.bounding_box.min_z == expected

    def test_top_elevation_for_unit_without_top(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        def drop_top(document: dict[str, Any]) -> None:
            del _section_model(document)["agsiModelElement"][0]["agsiGeometry"]["agsiGeometryTop"]

        edit_document(straight_wkt_dir, drop_top)
        with pytest.raises(AGSIInvalidDataError, match="has no top line"):
            _build(local_data_client, straight_wkt_dir)
        obj = _build(local_data_client, straight_wkt_dir, top_elevation=10.0)
        assert obj.surfaces[0].feature == "BoundarySurface"
        assert obj.surfaces[0].name == f"Top of {SURFICIAL}"
        assert obj.bounding_box.max_z == 10.0


class TestModels:
    def test_multiple_section_models(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        def duplicate(document: dict[str, Any]) -> None:
            second = copy.deepcopy(_section_model(document))
            second.update(modelID="Second", modelName="Second section")
            document["agsiModel"].append(second)

        edit_document(straight_wkt_dir, duplicate)
        result = get_geoscience_object_from_agsi(
            local_data_client, str(straight_wkt_dir), "unspecified", base_elevation=-50.0
        )
        assert isinstance(result, list)
        # The plan model only holds the alignment and is skipped.
        assert [obj.name for obj in result] == ["Silvertown mini NB section", "Second section"]

    def test_no_section_model(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        edit_document(straight_wkt_dir, lambda d: d.update(agsiModel=d["agsiModel"][1:]))
        with pytest.raises(AGSIUnsupportedGeometryError, match="no 2D section model"):
            _build(local_data_client, straight_wkt_dir)
        assert not local_data_client.save_table.called

    def test_model_without_alignment_uses_local_frame(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        edit_document(straight_wkt_dir, lambda d: _section_model(d).pop("alignmentID"))
        with caplog.at_level(logging.WARNING):
            obj = _build(local_data_client, straight_wkt_dir)
        assert "has no alignmentID" in caplog.text
        assert obj.sections[0].origin == [0.0, 0.0, 0.0]
        assert [folder.name for folder in obj.folders] == [obj.name]
        assert "alignmentID" not in obj.extensions["agsi"]

    def test_only_zero_thickness_areas(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        def flatten(document: dict[str, Any]) -> None:
            model = _section_model(document)
            surficial = model["agsiModelElement"][0]
            surficial["agsiGeometry"]["agsiGeometryBottom"] = surficial["agsiGeometry"]["agsiGeometryTop"]
            model["agsiModelElement"] = [surficial]

        edit_document(straight_wkt_dir, flatten)
        with pytest.raises(AGSIInvalidDataError, match="no section areas with a positive thickness"):
            _build(local_data_client, straight_wkt_dir)


class TestRejections:
    def test_curved_alignment(self, local_data_client: MagicMock, wkt_dir: Path) -> None:
        with pytest.raises(AGSIUnsupportedGeometryError, match="agsiModelAlignment 'NB'.*not straight"):
            _build(local_data_client, wkt_dir)
        assert not local_data_client.save_table.called

    def test_3d_profile(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        (straight_wkt_dir / "Profiles" / "NB_Alignment-01-Surficial-Top.txt").write_text(
            "LINESTRING (539591.028 179854.986 4.5, 539910.921 180226.890 4.0)"
        )
        with pytest.raises(AGSIUnsupportedGeometryError, match="3D"):
            _build(local_data_client, straight_wkt_dir)

    def test_missing_geometry_file(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        (straight_wkt_dir / "Profiles" / "NB_Alignment-02-Alluvium-Top.txt").unlink()
        with pytest.raises(AGSIDataFileIOError, match="NB_Alignment-02-Alluvium-Top.txt"):
            _build(local_data_client, straight_wkt_dir)

    def test_missing_alignment(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        edit_document(straight_wkt_dir, lambda d: d.update(agsiModel=d["agsiModel"][:1]))
        with pytest.raises(AGSIInvalidDataError, match="alignmentID 'NB' does not match"):
            _build(local_data_client, straight_wkt_dir)

    def test_unsupported_version(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        edit_document(straight_wkt_dir, lambda d: d["agsSchema"].update(version="1.1.0"))
        with pytest.raises(AGSIUnsupportedVersionError):
            _build(local_data_client, straight_wkt_dir)

    def test_missing_file(self, local_data_client: MagicMock, tmp_path: Path) -> None:
        with pytest.raises(AGSIDataFileIOError):
            _build(local_data_client, tmp_path / "missing.agsi")


class TestCoordinateReferenceSystem:
    def test_declared_system_is_not_applied(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            obj = get_geoscience_object_from_agsi(
                local_data_client, str(straight_wkt_dir), "unspecified", base_elevation=-50.0
            )
        assert isinstance(obj, GeologicalSections_V1_2_0)
        assert obj.coordinate_reference_system == "unspecified"
        assert "declares globalXYSystem 'EPSG:27700'" in caplog.text
        assert "It is not applied automatically" in caplog.text

    def test_given_crs_matching_declared(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _build(local_data_client, straight_wkt_dir)
        assert "globalXYSystem" not in caplog.text

    def test_given_crs_differs_from_declared(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            obj = get_geoscience_object_from_agsi(
                local_data_client, str(straight_wkt_dir), crs_from_epsg_code(32631), base_elevation=-50.0
            )
        assert isinstance(obj, GeologicalSections_V1_2_0)
        assert obj.coordinate_reference_system == Crs_V1_0_1_EpsgCode(epsg_code=32631)
        assert "EPSG:32631 differs from the AGSi globalXYSystem 'EPSG:27700'" in caplog.text

    def test_transformation_not_applied(
        self, local_data_client: MagicMock, straight_wkt_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def shift(document: dict[str, Any]) -> None:
            document["agsProject"]["agsProjectCoordinateSystem"][1]["transformShiftX"] = 100.0

        edit_document(straight_wkt_dir, shift)
        with caplog.at_level(logging.WARNING):
            obj = _build(local_data_client, straight_wkt_dir)
        assert "'BNG-XY' defines a transformation" in caplog.text
        assert obj.sections[0].origin[0] == START[0]
