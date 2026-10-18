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

"""Offline tests of the AGSi exporter: Evo objects come from the importer, clients are mocked."""

import dataclasses
import json
import logging
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evo_schemas.components import (
    BoolAttribute_V1_1_0,
    BoundingBox_V1_0_1,
    CategoryAttribute_V1_1_0,
    Crs_V1_0_1_OgcWkt,
    EmbeddedPolylineObject_V1_0_0,
    IntegerAttribute_V1_1_0,
    NanCategorical_V1_0_1,
    Rotation_V1_1_0,
    StringAttribute_V1_1_0,
)
from evo_schemas.elements import (
    BoolArray1_V1_0_1,
    FloatArray3_V1_0_1,
    IntegerArray1_V1_0_1,
    LookupTable_V1_0_1,
    StringArray_V1_0_1,
)
from evo_schemas.objects import Pointset_V1_2_0
from evo_schemas.objects.geological_sections import (
    GeologicalSections_V1_2_0,
    GeologicalSections_V1_2_0_Folder,
    GeologicalSections_V1_2_0_Folder_Items_SectionIndex,
)
from evo_schemas.objects.pointset import Pointset_V1_2_0_Locations
from pyproj import CRS

from conftest import edit_document
from evo.data_converters.agsi.exporter import (
    AGSIExporterException,
    SectionsSource,
    UnsupportedObjectError,
    UnsupportedSectionGeometryError,
    build_agsi_export,
    export_agsi,
)
from evo.data_converters.agsi.exporter.geological_sections import linestring_wkt
from evo.data_converters.agsi.importer.utils import get_geoscience_object_from_agsi
from evo.data_converters.agsi.reader import AgsiPackage
from evo.data_converters.common import EvoObjectMetadata, EvoWorkspaceMetadata
from evo.data_converters.common.crs import crs_from_epsg_code
from evo.data_converters.common.exceptions import MissingConnectionDetailsError

_CLIENTS = "evo.data_converters.agsi.exporter.evo_to_agsi.create_evo_object_service_and_data_client"
SURFICIAL = "Profile Surficial Alignment NB"
ALLUVIUM = "Profile Alluvium Alignment NB"
SAMPLE_DIR = os.environ.get("AGSI_SAMPLE_DIR")


def enrich(document: dict[str, Any]) -> None:
    """Add element names, colours and parameters (one with a caseID) to the mini section model."""
    surficial, alluvium = document["agsiModel"][0]["agsiModelElement"]
    surficial.update(elementName="Surficial deposits", colourRGB="#A0522D")
    surficial["agsiDataParameterValue"] = [
        {"codeID": "UnitWeightBulk", "valueNumeric": 18.5},
        {"codeID": "UndrainedShearStrength", "caseID": "Lower", "valueNumeric": 40.0},
        {"codeID": "Comment", "valueText": "Made ground"},
    ]
    alluvium["agsiDataParameterValue"] = [
        {"codeID": "UnitWeightBulk", "valueNumeric": 17.25},
        {"codeID": "UndrainedShearStrength", "caseID": "Lower", "valueNumeric": 0.1 + 0.2},
    ]


def import_mini(folder: Path, client: MagicMock, **kwargs: Any) -> GeologicalSections_V1_2_0:
    obj = get_geoscience_object_from_agsi(client, str(folder), crs_from_epsg_code(27700), **kwargs)
    assert isinstance(obj, GeologicalSections_V1_2_0)
    return obj


def loader(client: MagicMock) -> Any:
    return lambda component: pq.read_table(client.cache_location / str(component.data))


def export(objects: list[Any], client: MagicMock, path: Path, title: str = "Export") -> AgsiPackage:
    build_agsi_export([SectionsSource(obj, loader(client)) for obj in objects], title).write(path)
    return AgsiPackage.open(path)


def profile(package: AgsiPackage, geometry: Any) -> np.ndarray:
    lines = package.read_geometry(geometry).lines
    assert len(lines) == 1
    line: np.ndarray = lines[0]
    return line


def table(client: MagicMock, component: Any) -> np.ndarray:
    array: np.ndarray = loader(client)(component).to_pandas().to_numpy()
    return array


@pytest.fixture
def mini(straight_wkt_dir: Path, local_data_client: MagicMock) -> GeologicalSections_V1_2_0:
    edit_document(straight_wkt_dir, enrich)
    return import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)


class TestRoundTrip:
    def test_document(
        self, mini: GeologicalSections_V1_2_0, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        package = export([mini], local_data_client, tmp_path / "out.agsi", title="Round trip")
        source = AgsiPackage.open(straight_wkt_dir)
        assert package.schema_version == "1.0.1"
        document = json.loads(package.read_bytes(package.main_file).decode("utf-8"))
        assert document["agsSchema"]["version"] == "1.0.1"
        assert document["agsFile"]["title"] == document["agsProject"]["projectName"] == "Round trip"

        section, plan = package.models
        assert (section.model_id, section.model_name) == ("Silvertown-mini-NB-Section", "Silvertown mini NB section")
        assert section.coord_system_id == "NB-Section-XZ" and section.alignment_id == "NB"
        assert "Evo folder: NB alignment" in section.remarks
        systems = {system.system_id: system for system in package.document.ags_project.ags_project_coordinate_system}
        assert systems["NB-Section-XZ"].system_type == "XZ"
        xy = systems[plan.coord_system_id]
        assert (xy.system_type, xy.global_xy_system, xy.axis_units_xy) == ("XY", "EPSG:27700", "m")
        assert (xy.axis_name_x, xy.axis_name_y) == ("Easting", "Northing")

        (alignment,) = plan.agsi_model_alignment
        assert (alignment.alignment_id, alignment.alignment_name, alignment.start_chainage) == ("NB", "NB alignment", 0)
        assert package.read_bytes(alignment.agsi_geometry.file_uri) == (
            b"LINESTRING (539591.028 179854.986, 539910.921 180226.89)"
        )

        surficial, alluvium = section.agsi_model_element
        original = {element.element_id: element for element in source.models[0].agsi_model_element}
        assert [surficial.element_id, alluvium.element_id] == [SURFICIAL, ALLUVIUM]
        assert (surficial.element_name, surficial.colour_rgb, surficial.element_type) == (
            "Surficial deposits",
            "#A0522D",
            "Geological unit",
        )
        assert alluvium.element_name is None and alluvium.colour_rgb == "#FF7F0E"  # palette colour of the import
        for element in (surficial, alluvium):
            assert element.description == original[element.element_id].description
            assert element.geometry_object == "agsiGeometryAreaFromLines"
        # Profiles are written exactly, and the shared contact once.
        top, bottom = surficial.agsi_geometry.agsi_geometry_top, surficial.agsi_geometry.agsi_geometry_bottom
        assert alluvium.agsi_geometry.agsi_geometry_top.file_uri == bottom.file_uri
        assert alluvium.agsi_geometry.agsi_geometry_bottom is None
        source_surficial = original[SURFICIAL].agsi_geometry
        np.testing.assert_array_equal(profile(package, top), profile(source, source_surficial.agsi_geometry_top))
        np.testing.assert_array_equal(profile(package, bottom), profile(source, source_surficial.agsi_geometry_bottom))
        assert (top.geometry_type, top.file_format) == ("Profile", "WKT")
        assert section.agsi_model_boundary.bottom_elevation == -50.0
        assert section.agsi_model_boundary.top_elevation is None

        parameters = {
            (p.code_id, p.case_id): (p.value_numeric, p.value_text) for p in surficial.agsi_data_parameter_value
        }
        assert parameters == {
            ("UnitWeightBulk", None): (18.5, None),
            ("UndrainedShearStrength", "Lower"): (40.0, None),
            ("Comment", None): (None, "Made ground"),
        }
        assert [(p.code_id, p.case_id, p.value_numeric) for p in alluvium.agsi_data_parameter_value] == [
            ("UnitWeightBulk", None, 17.25),
            ("UndrainedShearStrength", "Lower", 0.1 + 0.2),
        ]
        with zipfile.ZipFile(tmp_path / "out.agsi") as archive:
            assert archive.testzip() is None
            names = archive.namelist()
        assert names[:2] == ["index.json", "Round_trip.json"]
        assert len(names) == 2 + 3  # alignment, top of Surficial, shared contact (base of Alluvium is a boundary)

    def test_reimport_is_identical(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.agsi"
        export([mini], local_data_client, path)
        again = import_mini(path, local_data_client)
        for name in ("vertices", "chunks"):
            np.testing.assert_array_equal(
                table(local_data_client, getattr(again.line_geometry, name)),
                table(local_data_client, getattr(mini.line_geometry, name)),
            )
        assert again.sections == mini.sections and again.volumes == mini.volumes
        # AGSi geometry descriptions are not kept in Evo, so contact descriptions come from the element names.
        assert [dataclasses.replace(s, description=None) for s in again.surfaces] == [
            dataclasses.replace(s, description=None) for s in mini.surfaces
        ]
        assert again.materials == mini.materials and again.layer_order == mini.layer_order
        assert again.extensions == mini.extensions
        assert again.bounding_box == mini.bounding_box
        assert [a.name for a in again.volume_attributes] == [a.name for a in mini.volume_attributes]
        for new, old in zip(again.volume_attributes, mini.volume_attributes):
            assert type(new) is type(old)
            assert loader(local_data_client)(new.values).equals(loader(local_data_client)(old.values))


TOP_FILE = "NB_Alignment-01-Surficial-Top.txt"
CONTACT_FILES = ("NB_Alignment-01-Surficial-Base.txt", "NB_Alignment-02-Alluvium-Top.txt")


def read_profile(path: Path) -> np.ndarray:
    text = path.read_text()
    return np.array([[float(value) for value in point.split()] for point in text[text.index("(") + 1 : -1].split(",")])


def pinch(folder: Path, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Make the base of Surficial (= top of Alluvium) meet its top at ``indices``; return both lines."""
    top = read_profile(folder / "Profiles" / TOP_FILE)
    contact = read_profile(folder / "Profiles" / CONTACT_FILES[0])
    contact[indices, 1] = top[indices, 1]
    for name in CONTACT_FILES:
        (folder / "Profiles" / name).write_text(linestring_wkt(contact))
    return top, contact


def with_historical_surfaces(obj: GeologicalSections_V1_2_0, entries: Any) -> GeologicalSections_V1_2_0:
    """Add ``extensions.agsi.surfaces`` as written by earlier versions of the importer."""
    return dataclasses.replace(obj, extensions={"agsi": {**obj.extensions["agsi"], "surfaces": entries}})


def package_bytes(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def main_document(files: dict[str, bytes]) -> Any:
    return json.loads(files[json.loads(files["index.json"])["agsiIndex"]["filePath"]])


def assert_stable(path: Path, client: MagicMock, tmp_path: Path) -> GeologicalSections_V1_2_0:
    """Importing an exported package and exporting it again changes nothing after the first time.

    The second export only drops the remarks the exporter wrote on ``"(part n)"`` elements (element
    remarks are not kept in Evo); a third export is byte-identical to the second.
    """
    again = import_mini(path, client)
    assert "surfaces" not in again.extensions["agsi"]
    export([again], client, tmp_path / "second.agsi")
    first, second = package_bytes(path), package_bytes(tmp_path / "second.agsi")
    expected = main_document(first)
    for model in expected["agsiModel"]:
        for element in model.get("agsiModelElement", []):
            if element.get("remarks", "").startswith("One of several separate polygons of"):
                del element["remarks"]
    assert main_document(second) == expected
    main = json.loads(first["index.json"])["agsiIndex"]["filePath"]
    assert {k: v for k, v in second.items() if k != main} == {k: v for k, v in first.items() if k != main}
    export([import_mini(tmp_path / "second.agsi", client)], client, tmp_path / "third.agsi")
    assert package_bytes(tmp_path / "third.agsi") == second
    return again


def assert_same_import(again: GeologicalSections_V1_2_0, obj: GeologicalSections_V1_2_0, client: MagicMock) -> None:
    """The re-imported object has the same line geometry, volumes, surfaces and extensions."""
    for name in ("vertices", "chunks"):
        assert loader(client)(getattr(again.line_geometry, name)).equals(
            loader(client)(getattr(obj.line_geometry, name))
        )
    assert again.sections == obj.sections and again.volumes == obj.volumes
    assert [dataclasses.replace(s, description=None) for s in again.surfaces] == [
        dataclasses.replace(s, description=None) for s in obj.surfaces
    ]
    assert again.extensions == obj.extensions


WHOLE = slice(None)


class TestElementGrouping:
    """A unit that pinches out is several Evo volumes; they are one element only where they touch."""

    @pytest.mark.parametrize(
        ("indices", "volumes", "elements"),
        [
            ([2], 2, {"": WHOLE}),  # internal pinch at one vertex: the volumes touch and are merged
            ([2, 3], 2, {" (part 1)": slice(0, 3), " (part 2)": slice(3, None)}),  # internal zero-thickness run
            ([0], 1, {"": WHOLE}),  # left end pinched
            ([0, 5], 1, {"": WHOLE}),  # both ends pinched
            ([4, 5], 1, {"": slice(0, 5)}),  # zero-thickness run at the right end: the lines end with the volume
            ([0, 2, 3, 5], 2, {" (part 1)": slice(0, 3), " (part 2)": slice(3, None)}),
        ],
    )
    def test_pinched_unit(
        self,
        straight_wkt_dir: Path,
        local_data_client: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        indices: list[int],
        volumes: int,
        elements: dict[str, slice],
    ) -> None:
        top, contact = pinch(straight_wkt_dir, indices)
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        assert Counter(v.layer for v in obj.volumes) == {SURFICIAL: volumes, ALLUVIUM: 1}

        path = tmp_path / "out.agsi"
        with caplog.at_level(logging.WARNING):
            package = export([obj], local_data_client, path)
        assert ("'(part n)' element" in caplog.text) == (len(elements) > 1)
        *parts, alluvium = package.models[0].agsi_model_element
        assert [e.element_id for e in parts] == [SURFICIAL + suffix for suffix in elements]
        # Each part has a run of the source lines; nothing is added between or beyond the volumes.
        for element, run in zip(parts, elements.values()):
            assert (element.remarks is not None) == (len(elements) > 1)
            geometry = element.agsi_geometry
            np.testing.assert_array_equal(profile(package, geometry.agsi_geometry_top), top[run])
            np.testing.assert_array_equal(profile(package, geometry.agsi_geometry_bottom), contact[run])
        np.testing.assert_array_equal(profile(package, alluvium.agsi_geometry.agsi_geometry_top), contact)
        shared = (
            alluvium.agsi_geometry.agsi_geometry_top.file_uri == parts[0].agsi_geometry.agsi_geometry_bottom.file_uri
        )
        assert shared == (list(elements.values()) == [WHOLE])

        again = assert_stable(path, local_data_client, tmp_path)
        if list(elements.values()) == [WHOLE]:
            assert_same_import(again, obj, local_data_client)

    def test_unit_touching_generated_base(
        self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        # Alluvium has no bottom line; its top touches the model base inside the section, so it is
        # two volumes over a base sampled at its chainages, and one element with no bottom again.
        contact = read_profile(straight_wkt_dir / "Profiles" / CONTACT_FILES[0])
        contact[2, 1] = 1.0
        for name in CONTACT_FILES:
            (straight_wkt_dir / "Profiles" / name).write_text(linestring_wkt(contact))
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=1.0)
        assert Counter(v.layer for v in obj.volumes)[ALLUVIUM] == 2
        package = export([obj], local_data_client, tmp_path / "out.agsi")
        section = package.models[0]
        _, alluvium = section.agsi_model_element
        assert alluvium.agsi_geometry.agsi_geometry_bottom is None
        assert section.agsi_model_boundary.bottom_elevation == 1.0
        np.testing.assert_array_equal(profile(package, alluvium.agsi_geometry.agsi_geometry_top), contact)
        assert_same_import(import_mini(tmp_path / "out.agsi", local_data_client), obj, local_data_client)

    def test_edited_geometry_is_exported(
        self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        # Moving a shared vertex in Evo changes the surface and the volume together.
        top, _ = pinch(straight_wkt_dir, [2])
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        vertices = table(local_data_client, obj.line_geometry.vertices).copy()
        (row,) = np.flatnonzero((vertices == top[1]).all(axis=1))
        vertices[row, 1] += 0.5
        saved = local_data_client.save_table(pa.table({"u": vertices[:, 0], "v": vertices[:, 1]}))
        edited = dataclasses.replace(
            obj,
            line_geometry=dataclasses.replace(
                obj.line_geometry, vertices=dataclasses.replace(obj.line_geometry.vertices, **saved)
            ),
        )
        package = export([edited], local_data_client, tmp_path / "out.agsi")
        surficial, _ = package.models[0].agsi_model_element
        exported = profile(package, surficial.agsi_geometry.agsi_geometry_top)
        top[1, 1] += 0.5
        np.testing.assert_array_equal(exported, top)

    def test_deleted_volume_exports_remaining_volumes(
        self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        top, contact = pinch(straight_wkt_dir, [2])
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        volumes = [obj.volumes[0], *obj.volumes[2:]]
        section = dataclasses.replace(
            obj.sections[0], volumes=dataclasses.replace(obj.sections[0].volumes, count=len(volumes))
        )
        edited = dataclasses.replace(obj, volumes=volumes, sections=[section], volume_attributes=[])
        package = export([edited], local_data_client, tmp_path / "out.agsi")
        surficial, alluvium = package.models[0].agsi_model_element
        assert [surficial.element_id, alluvium.element_id] == [SURFICIAL, ALLUVIUM]
        np.testing.assert_array_equal(profile(package, surficial.agsi_geometry.agsi_geometry_top), top[:3])
        np.testing.assert_array_equal(profile(package, surficial.agsi_geometry.agsi_geometry_bottom), contact[:3])

    def test_volumes_of_one_element_that_differ_are_parts(
        self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        pinch(straight_wkt_dir, [2])
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        renamed = [obj.volumes[0], dataclasses.replace(obj.volumes[1], name="Other"), *obj.volumes[2:]]
        package = export([dataclasses.replace(obj, volumes=renamed)], local_data_client, tmp_path / "out.agsi")
        assert [e.element_id for e in package.models[0].agsi_model_element] == [
            f"{SURFICIAL} (part 1)",
            f"{SURFICIAL} (part 2)",
            ALLUVIUM,
        ]

    @pytest.mark.parametrize("indices", [[2], [2, 3]])
    def test_historical_surfaces_extension_is_ignored(
        self,
        straight_wkt_dir: Path,
        local_data_client: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        indices: list[int],
    ) -> None:
        pinch(straight_wkt_dir, indices)
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        export([obj], local_data_client, tmp_path / "plain.agsi")
        legacy = [  # as written by earlier importers: complete top and bottom of each element
            {"name": surface.name, "topOf": top_of, "bottomOf": bottom_of}
            for surface, top_of, bottom_of in zip(
                obj.surfaces, [[SURFICIAL], [ALLUVIUM], []], [[], [SURFICIAL], [ALLUVIUM]]
            )
        ]
        for number, entries in enumerate([legacy, legacy[:1], "not a list"]):
            path = tmp_path / f"historical-{number}.agsi"
            with caplog.at_level(logging.INFO):
                export([with_historical_surfaces(obj, entries)], local_data_client, path)
            assert "extensions.agsi.surfaces (top/bottom relationships recorded by earlier" in caplog.text
            assert package_bytes(path) == package_bytes(tmp_path / "plain.agsi")

    def test_zero_thickness_element_is_not_exported(
        self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        pinch(straight_wkt_dir, list(range(6)))
        obj = import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)
        assert [v.layer for v in obj.volumes] == [ALLUVIUM]
        package = export([obj], local_data_client, tmp_path / "out.agsi")
        assert [e.element_id for e in package.models[0].agsi_model_element] == [ALLUVIUM]


class TestMapping:
    def test_attribute_types(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        save = local_data_client.save_table
        category = CategoryAttribute_V1_1_0(
            name="Stratum",
            key="Stratum",
            nan_description=NanCategorical_V1_0_1(values=[-1]),
            table=LookupTable_V1_0_1(
                **save(pa.table({"key": pa.array([1, 2], pa.int32()), "value": pa.array(["Made", "Clay"])}))
            ),
            values=IntegerArray1_V1_0_1(**save(pa.table({"values": pa.array([2, -1], pa.int32())}))),
        )
        integer = IntegerAttribute_V1_1_0(
            name="Count",
            key="Count",
            nan_description=NanCategorical_V1_0_1(values=[-99]),
            values=IntegerArray1_V1_0_1(**save(pa.table({"values": pa.array([-99, 7], pa.int32())}))),
        )
        flag = BoolAttribute_V1_1_0(
            name="Flag", key="Flag", values=BoolArray1_V1_0_1(**save(pa.table({"values": pa.array([True, False])})))
        )
        obj = dataclasses.replace(mini, volume_attributes=[*mini.volume_attributes, category, integer, flag])
        package = export([obj], local_data_client, tmp_path / "out.agsi")
        surficial, alluvium = package.models[0].agsi_model_element
        extra = [(p.code_id, p.value_numeric, p.value_text) for p in surficial.agsi_data_parameter_value[3:]]
        assert extra == [("Stratum", None, "Clay")]
        extra = [(p.code_id, p.value_numeric, p.value_text) for p in alluvium.agsi_data_parameter_value[2:]]
        assert extra == [("Count", 7.0, None)]

    def test_attribute_length_mismatch(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        values = StringArray_V1_0_1(**local_data_client.save_table(pa.table({"values": pa.array(["a"])})))
        attribute = StringAttribute_V1_1_0(name="Short", key="Short", values=values)
        obj = dataclasses.replace(mini, volume_attributes=[*mini.volume_attributes, attribute])
        with pytest.raises(AGSIExporterException, match="1 values for 2 volumes"):
            export([obj], local_data_client, tmp_path / "out.agsi")

    def test_multiple_sections_and_nested_folders(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        moved = dataclasses.replace(mini.sections[0], name="South", origin=[539600.0, 179800.0, 5.0])
        folders = [
            GeologicalSections_V1_2_0_Folder(
                name="Project",
                items=[
                    GeologicalSections_V1_2_0_Folder(
                        name="North line", items=[GeologicalSections_V1_2_0_Folder_Items_SectionIndex(section_index=0)]
                    ),
                    GeologicalSections_V1_2_0_Folder_Items_SectionIndex(section_index=1),
                ],
            )
        ]
        obj = dataclasses.replace(mini, name="Two sections", sections=[mini.sections[0], moved], folders=folders)
        package = export([obj, mini], local_data_client, tmp_path / "out.agsi")

        first, second, plan, mini_section, mini_plan = package.models
        # The AGSi identifiers of the extension describe a single section, so they are not used here.
        assert [first.model_id, second.model_id] == [
            "Two sections - Silvertown mini NB section",
            "Two sections - South",
        ]
        assert mini_section.model_id == "Silvertown-mini-NB-Section"
        names = {a.alignment_id: a.alignment_name for a in plan.agsi_model_alignment}
        assert names == {first.alignment_id: "North line", second.alignment_id: "Project"}
        assert "Evo folder: Project / North line." in first.remarks
        assert plan.coord_system_id == mini_plan.coord_system_id == "Evo-XY"  # one XY system per CRS
        # Elevations include the section origin.
        offset = [profile(package, m.agsi_model_element[0].agsi_geometry.agsi_geometry_top) for m in (first, second)]
        np.testing.assert_array_equal(offset[1][:, 1], offset[0][:, 1] + 5.0)
        assert second.agsi_model_boundary.bottom_elevation == -45.0
        south = next(a for a in plan.agsi_model_alignment if a.alignment_id == second.alignment_id)
        assert package.read_bytes(south.agsi_geometry.file_uri).startswith(b"LINESTRING (539600 179800, ")

    def test_layer_order_and_untouched_boundary(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        # Without layer_order the volume order is kept; boundary surfaces are not needed to export.
        obj = dataclasses.replace(
            mini,
            layer_order=list(reversed(mini.layer_order)),
            sections=[dataclasses.replace(mini.sections[0], surfaces=None)],
        )
        package = export([obj], local_data_client, tmp_path / "out.agsi")
        section = package.models[0]
        assert [e.element_id for e in section.agsi_model_element] == [ALLUVIUM, SURFICIAL]
        # Not recognised as generated: the base of Alluvium is written as a profile.
        assert section.agsi_model_boundary is None
        base = profile(package, section.agsi_model_element[0].agsi_geometry.agsi_geometry_bottom)
        assert base.tolist() == [[0.0, -50.0], [22.975514267068423, -50.0]]

    @pytest.mark.parametrize(
        ("crs", "units", "global_system"),
        [
            (Crs_V1_0_1_OgcWkt(ogc_wkt=CRS.from_epsg(2264).to_wkt()), "ftUS", None),
            ("unspecified", "unknown", None),
        ],
    )
    def test_coordinate_systems(
        self,
        mini: GeologicalSections_V1_2_0,
        local_data_client: MagicMock,
        tmp_path: Path,
        crs: Any,
        units: str,
        global_system: Optional[str],
    ) -> None:
        obj = dataclasses.replace(mini, coordinate_reference_system=crs)
        package = export([obj], local_data_client, tmp_path / "out.agsi")
        systems = {s.system_id: s for s in package.document.ags_project.ags_project_coordinate_system}
        xy, xz = systems["Evo-XY"], systems["NB-Section-XZ"]
        assert xy.global_xy_system == global_system
        assert xy.axis_units_xy == xz.axis_units_xy == xz.axis_units_z == units
        if isinstance(crs, Crs_V1_0_1_OgcWkt):
            assert crs.ogc_wkt in xy.remarks


class TestCodeSets:
    """Interim code sets: the exporter defines every codeID and caseID it writes."""

    @staticmethod
    def code_sets(package: AgsiPackage) -> dict[str, list[dict[str, Any]]]:
        project = package.document.ags_project
        sets = project.ags_project_code_set if project else []
        assert all(s["usedByObject"] == "agsiDataParameterValue" for s in sets)
        by_attribute = {s["usedByAttribute"]: s["agsProjectCode"] for s in sets}
        assert len(by_attribute) == len(sets)
        return by_attribute

    @classmethod
    def assert_resolved(cls, package: AgsiPackage) -> dict[str, list[str]]:
        """Every codeID and caseID written is defined once in the code set of its attribute."""
        sets = cls.code_sets(package)
        defined = {attribute: [c["codeID"] for c in codes] for attribute, codes in sets.items()}
        for attribute, codes in defined.items():
            assert len(codes) == len(set(codes))
        used: dict[str, set[str]] = {"codeID": set(), "caseID": set()}
        for model in package.models:
            for element in model.agsi_model_element:
                for parameter in element.agsi_data_parameter_value:
                    used["codeID"].add(parameter.code_id)
                    if parameter.case_id is not None:
                        used["caseID"].add(parameter.case_id)
        assert {a: set(c) for a, c in defined.items()} == {a: c for a, c in used.items() if c}
        return defined

    def test_deduplicated_across_elements(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        package = export([mini], local_data_client, tmp_path / "out.agsi")
        assert self.code_sets(package) == {
            "codeID": [
                {"codeID": "UnitWeightBulk", "description": "UnitWeightBulk"},
                {"codeID": "UndrainedShearStrength", "description": "UndrainedShearStrength"},
                {"codeID": "Comment", "description": "Comment"},
            ],
            "caseID": [{"codeID": "Lower", "description": "Lower"}],
        }
        self.assert_resolved(package)
        (code_set, case_set) = package.document.ags_project.ags_project_code_set
        assert "units" not in json.dumps(code_set)
        assert code_set["remarks"] == case_set["remarks"]

    def test_shared_across_sections_and_objects(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        moved = dataclasses.replace(mini.sections[0], name="South", origin=[539600.0, 179800.0, 5.0])
        two = dataclasses.replace(mini, name="Two sections", sections=[mini.sections[0], moved])
        package = export([two, mini], local_data_client, tmp_path / "out.agsi")
        assert len([m for m in package.models if m.agsi_model_element]) == 3
        assert self.assert_resolved(package) == {
            "codeID": ["UnitWeightBulk", "UndrainedShearStrength", "Comment"],
            "caseID": ["Lower"],
        }

    def test_no_parameters(self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path) -> None:
        package = export(
            [import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)],
            local_data_client,
            tmp_path / "out.agsi",
        )
        assert all(not e.agsi_data_parameter_value for e in package.models[0].agsi_model_element)
        assert package.document.ags_project.ags_project_code_set == []
        assert b"agsProjectCodeSet" not in package.read_bytes(package.main_file)

    def test_codes_without_cases(self, straight_wkt_dir: Path, local_data_client: MagicMock, tmp_path: Path) -> None:
        def no_cases(document: dict[str, Any]) -> None:
            enrich(document)
            for element in document["agsiModel"][0]["agsiModelElement"]:
                for parameter in element["agsiDataParameterValue"]:
                    parameter.pop("caseID", None)

        edit_document(straight_wkt_dir, no_cases)
        package = export(
            [import_mini(straight_wkt_dir, local_data_client, base_elevation=-50.0)],
            local_data_client,
            tmp_path / "out.agsi",
        )
        assert list(self.assert_resolved(package)) == ["codeID"]

    def test_unusual_codes_and_skipped_values(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        save = local_data_client.save_table

        def text(name: str, values: list[Optional[str]]) -> StringAttribute_V1_1_0:
            return StringAttribute_V1_1_0(
                name=name, key=name, values=StringArray_V1_0_1(**save(pa.table({"values": pa.array(values)})))
            )

        unusual = ' Effective cohesion c\' (kPa) / ü "x" $..[*] '
        flag = BoolAttribute_V1_1_0(
            name="Flag", key="Flag", values=BoolArray1_V1_0_1(**save(pa.table({"values": pa.array([True, False])})))
        )
        attributes = [
            *mini.volume_attributes,
            text(unusual, [None, "a"]),
            text("unitweightbulk", ["b", "c"]),  # differs from UnitWeightBulk only in case
            StringAttribute_V1_1_0(
                name="All missing",
                key="All missing",
                values=StringArray_V1_0_1(**save(pa.table({"values": pa.array([None, None], pa.string())}))),
            ),
            flag,
        ]
        package = export(
            [dataclasses.replace(mini, volume_attributes=attributes)], local_data_client, tmp_path / "out.agsi"
        )
        # Codes are kept exactly; only codes of values written are defined.
        assert self.assert_resolved(package)["codeID"] == [
            "UnitWeightBulk",
            "UndrainedShearStrength",
            "Comment",
            "unitweightbulk",
            unusual,
        ]


class TestErrors:
    def test_non_vertical_section(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        tilted = Rotation_V1_1_0(dip_azimuth=130.0, dip=60.0, pitch=180.0)
        obj = dataclasses.replace(mini, sections=[dataclasses.replace(mini.sections[0], rotation=tilted)])
        with pytest.raises(UnsupportedSectionGeometryError, match="not a vertical plane"):
            export([obj], local_data_client, tmp_path / "out.agsi")
        assert not (tmp_path / "out.agsi").exists()

    def test_internal_rings(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        hole = EmbeddedPolylineObject_V1_0_0(name="hole", parts=mini.volumes[0].parts)
        volumes = [dataclasses.replace(mini.volumes[0], internal_rings=[hole]), mini.volumes[1]]
        with pytest.raises(UnsupportedSectionGeometryError, match="holes"):
            export([dataclasses.replace(mini, volumes=volumes)], local_data_client, tmp_path / "out.agsi")

    def test_overhanging_volume(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        # Make the polygon of Surficial run back over itself by reusing its top as the right edge.
        parts = mini.volumes[0].parts
        bad = dataclasses.replace(mini.volumes[0], parts=[parts[0], dataclasses.replace(parts[0], reversed=True)])
        with pytest.raises(UnsupportedSectionGeometryError, match="zero area"):
            export([dataclasses.replace(mini, volumes=[bad, mini.volumes[1]])], local_data_client, tmp_path / "x.agsi")

    def test_disconnected_volume_parts(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        bad = dataclasses.replace(mini.volumes[0], parts=list(reversed(mini.volumes[0].parts)))
        with pytest.raises(AGSIExporterException, match="connected"):
            export([dataclasses.replace(mini, volumes=[bad, mini.volumes[1]])], local_data_client, tmp_path / "x.agsi")

    def test_object_without_section_volumes(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        empty = dataclasses.replace(mini.sections[0], volumes=None, surfaces=None)
        with pytest.raises(AGSIExporterException, match="no section volumes"):
            export([dataclasses.replace(mini, sections=[empty])], local_data_client, tmp_path / "x.agsi")

    def test_empty_section_is_skipped(
        self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path
    ) -> None:
        empty = dataclasses.replace(mini.sections[0], name="Empty", volumes=None, surfaces=None)
        package = export(
            [dataclasses.replace(mini, sections=[empty, *mini.sections])], local_data_client, tmp_path / "x.agsi"
        )
        assert [m.model_id for m in package.models][0] == "Silvertown mini NB section - Silvertown mini NB section"

    def test_nothing_to_export(self) -> None:
        with pytest.raises(AGSIExporterException, match="No objects"):
            build_agsi_export([], "x")


class TestExportAgsi:
    OBJECT_ID = UUID("00000000-0000-0000-0000-000000000001")

    def clients(self, document: dict[str, Any], local_data_client: MagicMock) -> tuple[MagicMock, MagicMock]:
        service_client = MagicMock()
        service_client.download_object_by_id = AsyncMock(return_value=SimpleNamespace(as_dict=lambda: document))
        data_client = MagicMock()
        cache = local_data_client.cache_location
        data_client.download_table = AsyncMock(
            side_effect=lambda object_id, version_id, table_info: pq.read_table(cache / table_info["data"])
        )
        return service_client, data_client

    @pytest.mark.usefixtures("block_network")
    def test_export(self, mini: GeologicalSections_V1_2_0, local_data_client: MagicMock, tmp_path: Path) -> None:
        service_client, data_client = self.clients(mini.as_dict(), local_data_client)
        workspace = EvoWorkspaceMetadata(org_id="org", workspace_id="ws", hub_url="https://example.invalid")
        path = tmp_path / "Exported sections.agsi"
        with patch(_CLIENTS, return_value=(service_client, data_client)) as create_clients:
            export_agsi(
                path, [EvoObjectMetadata(object_id=self.OBJECT_ID, version_id="7")], evo_workspace_metadata=workspace
            )

        create_clients.assert_called_once_with(evo_workspace_metadata=workspace, service_manager_widget=None)
        service_client.download_object_by_id.assert_awaited_once_with(self.OBJECT_ID, "7")
        assert data_client.download_table.await_count >= 4
        assert {call.args[:2] for call in data_client.download_table.await_args_list} == {(self.OBJECT_ID, "7")}
        package = AgsiPackage.open(path)
        assert package.document.ags_file.title == "Exported sections"
        assert f"ID {self.OBJECT_ID}, version 7" in package.document.ags_file.description
        assert [e.element_id for e in package.models[0].agsi_model_element] == [SURFICIAL, ALLUVIUM]

    def test_unknown_schema(self, local_data_client: MagicMock, tmp_path: Path) -> None:
        document = {"schema": "/objects/not-a-schema/1.0.0/not-a-schema.schema.json"}
        with patch(_CLIENTS, return_value=self.clients(document, local_data_client)):
            with pytest.raises(UnsupportedObjectError, match="Unknown Geoscience Object schema"):
                export_agsi(tmp_path / "x.agsi", [EvoObjectMetadata(object_id=self.OBJECT_ID)], MagicMock())
        assert not (tmp_path / "x.agsi").exists()

    def test_unsupported_object(self, local_data_client: MagicMock, tmp_path: Path) -> None:
        pointset = Pointset_V1_2_0(
            name="points",
            uuid=None,
            bounding_box=BoundingBox_V1_0_1(min_x=0.0, max_x=0.0, min_y=0.0, max_y=0.0, min_z=0.0, max_z=0.0),
            coordinate_reference_system="unspecified",
            locations=Pointset_V1_2_0_Locations(
                coordinates=FloatArray3_V1_0_1(data="0" * 64, length=1, width=3, data_type="float64")
            ),
        )
        with patch(_CLIENTS, return_value=self.clients(pointset.as_dict(), local_data_client)):
            with pytest.raises(UnsupportedObjectError, match="Pointset_V1_2_0 .* not supported"):
                export_agsi(tmp_path / "x.agsi", [EvoObjectMetadata(object_id=self.OBJECT_ID)], MagicMock())

    def test_no_objects(self, tmp_path: Path) -> None:
        with pytest.raises(AGSIExporterException, match="No objects"):
            export_agsi(tmp_path / "x.agsi", [], MagicMock())

    def test_missing_connection_details(self, tmp_path: Path) -> None:
        with pytest.raises(MissingConnectionDetailsError):
            export_agsi(tmp_path / "x.agsi", [EvoObjectMetadata(object_id=self.OBJECT_ID)])


PART = re.compile(r" \(part \d+\)$")


def run_of(line: np.ndarray, source: np.ndarray) -> bool:
    """Whether ``line`` is a run of consecutive vertices of ``source``."""
    starts = np.flatnonzero((source == line[0]).all(axis=1))
    return any(np.array_equal(source[start : start + len(line)], line) for start in starts)


@pytest.mark.skipif(not SAMPLE_DIR, reason="Set AGSI_SAMPLE_DIR to the folder of the converted AGSi 1.0.1 samples")
def test_straight_synthetic_sample_round_trip(local_data_client: MagicMock, tmp_path: Path) -> None:
    assert SAMPLE_DIR is not None
    path = Path(SAMPLE_DIR) / "Silvertown2D_wkt_straight_synthetic.agsi"
    obj = import_mini(path, local_data_client, base_elevation=-100.0)
    # Units that pinch out are several (valid, simple) Evo volumes: 22 volumes for 11 elements.
    assert (len(obj.volumes), len(obj.surfaces)) == (22, 12)
    assert "surfaces" not in obj.extensions["agsi"]
    first = tmp_path / "first.agsi"
    exported = export([obj], local_data_client, first)

    source = AgsiPackage.open(path)
    (section,) = [model for model in source.models if model.alignment_id]
    original = {e.element_id: e for e in section.agsi_model_element}
    (exported_section, _) = exported.models
    assert (exported_section.model_id, exported_section.alignment_id) == (section.model_id, section.alignment_id)
    elements = exported_section.agsi_model_element

    # Separate volumes of one unit are separate '(part n)' elements; touching volumes are merged.
    bases = [PART.sub("", e.element_id) for e in elements]
    assert list(dict.fromkeys(bases)) == list(original)
    assert {base: n for base, n in Counter(bases).items() if n > 1} == {
        "Profile Surficial Alignment NB": 4,
        "Profile Alluvium Alignment NB": 5,
        "Profile River Terrace Deposits Alignment NB": 2,
        "Profile Woolwich Formation Lower Shelly Clay Alignment NB": 2,
        "Profile Upnor Formation Alignment NB": 2,
    }
    assert len(elements) == 21
    # The codes in use are defined again (descriptions are not kept in Evo, so they repeat the code).
    source_codes = {c["codeID"] for s in source.document.ags_project.ags_project_code_set for c in s["agsProjectCode"]}
    assert set(TestCodeSets.assert_resolved(exported)["codeID"]) == source_codes
    for base, new in zip(bases, elements):
        old = original[base]
        assert (new.element_type, new.description, new.geometry_object) == (
            old.element_type,
            old.description,
            old.geometry_object,
        )
        assert (new.remarks is not None) == (new.element_id != base)
        assert [p.model_dump(exclude_none=True) for p in new.agsi_data_parameter_value] == [
            p.model_dump(exclude_none=True) for p in old.agsi_data_parameter_value
        ]
        # Every profile is a run of the source line: no geometry is invented.
        for which in ("agsi_geometry_top", "agsi_geometry_bottom"):
            old_geometry, new_geometry = getattr(old.agsi_geometry, which), getattr(new.agsi_geometry, which)
            if old_geometry is None:
                assert new_geometry is None  # Chalk: top only, down to the model boundary
                continue
            assert run_of(profile(exported, new_geometry), profile(source, old_geometry))
    with zipfile.ZipFile(first) as archive:
        profiles = [name for name in archive.namelist() if name.startswith("Profiles/")]
    assert len(profiles) == 40
    assert exported_section.agsi_model_boundary.bottom_elevation == -100.0

    # The split package is stable: importing and exporting it again changes nothing.
    again = assert_stable(first, local_data_client, tmp_path)
    assert (len(again.volumes), len(again.surfaces), len(again.layer_order)) == (22, 41, 21)
    # With or without relationships recorded by earlier importers, the export is the same.
    legacy = [{"name": s.name, "topOf": [], "bottomOf": []} for s in obj.surfaces]
    export([with_historical_surfaces(obj, legacy)], local_data_client, tmp_path / "legacy.agsi")
    assert package_bytes(tmp_path / "legacy.agsi") == package_bytes(first)
