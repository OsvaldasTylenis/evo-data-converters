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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import zip_folder

from evo.data_converters.agsi.reader import (
    AGSIDataFileIOError,
    AGSIInvalidDataError,
    AGSIUnsupportedVersionError,
    AgsiGeometryAreaFromLines,
    AgsiPackage,
    GeometryFileFormat,
    is_supported_version,
)

MAIN_FILE = "Silvertown_mini.json"
V101_SCHEMA = {"name": "AGSi", "version": "1.0.1"}


def _write_zip(path: Path, members: dict[str, object]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content if isinstance(content, (str, bytes)) else json.dumps(content))
    return path


def _section_lines(package: AgsiPackage) -> dict[tuple[str, str], np.ndarray]:
    section = package.models[0]
    return {
        (str(element.element_id), role): package.read_geometry(geometry).single_line()
        for element in section.agsi_model_element
        for role, geometry in element.geometry_files()
    }


def _assert_mini(package: AgsiPackage, file_format: GeometryFileFormat = GeometryFileFormat.WKT) -> None:
    assert package.schema_version == "1.0.1"
    document = package.document
    assert document.ags_file is not None and document.ags_file.title == "Silvertown mini"

    section, plan = package.models
    surficial, alluvium = section.agsi_model_element
    assert all(isinstance(e.agsi_geometry, AgsiGeometryAreaFromLines) for e in (surficial, alluvium))
    assert [role for role, _ in surficial.geometry_files()] == ["top", "bottom"]
    assert [role for role, _ in alluvium.geometry_files()] == ["top"]
    for _, geometry in surficial.geometry_files() + alluvium.geometry_files():
        assert geometry.file_format == file_format.value
        assert package.read_geometry(geometry).source_format is file_format

    lines = _section_lines(package)
    for line in lines.values():
        assert line.shape == (6, 2)  # chainage, elevation
    top = lines[("Profile Surficial Alignment NB", "top")]
    np.testing.assert_array_equal(top[:2], [[0.0, 4.618489312689358], [4.543003956140253, 4.515691137556667]])
    # Base of one unit is the top of the next.
    np.testing.assert_array_equal(
        lines[("Profile Surficial Alignment NB", "bottom")], lines[("Profile Alluvium Alignment NB", "top")]
    )
    assert package.warnings == []


class TestOpen:
    def test_zip(self, wkt_zip: Path) -> None:
        package = AgsiPackage.open(wkt_zip)
        assert package.is_zip and package.main_file == MAIN_FILE
        _assert_mini(package)
        assert sorted(package.list_files()) == [
            "Alignments/NB_Alignment.txt",
            "Profiles/NB_Alignment-01-Surficial-Base.txt",
            "Profiles/NB_Alignment-01-Surficial-Top.txt",
            "Profiles/NB_Alignment-02-Alluvium-Top.txt",
            MAIN_FILE,
            "index.json",
        ]

    def test_dxf_zip(self, dxf_zip: Path) -> None:
        _assert_mini(AgsiPackage.open(dxf_zip), GeometryFileFormat.DXF)

    def test_zip_with_any_extension(self, wkt_dir: Path, tmp_path: Path) -> None:
        _assert_mini(AgsiPackage.open(zip_folder(wkt_dir, tmp_path / "package.zip")))

    @pytest.mark.parametrize("relative", ["", "index.json", MAIN_FILE])
    def test_folder_index_or_main_json(self, wkt_dir: Path, relative: str) -> None:
        package = AgsiPackage.open(wkt_dir / relative if relative else wkt_dir)
        assert not package.is_zip
        _assert_mini(package)

    def test_v1_bare_json(self, v1_dir: Path) -> None:
        package = AgsiPackage.open(str(v1_dir / "model.json"))
        assert package.schema_version == "1.0.1"
        assert package.index == {}
        model = package.models[0]
        surface = package.read_geometry(model.agsi_model_element[0].geometry_files()[0][1])
        assert surface.srid == 27700
        assert surface.to_triangle_mesh().triangles.shape == (2, 3)
        alignment = model.agsi_model_alignment[0].agsi_geometry
        assert alignment is not None
        assert package.read_geometry(alignment).single_line().shape == (3, 2)

    def test_v1_folder_and_zip_without_index(self, v1_dir: Path, tmp_path: Path) -> None:
        for path in (v1_dir, zip_folder(v1_dir, tmp_path / "v1.agsi")):
            package = AgsiPackage.open(path)
            assert package.main_file == "model.json"
            assert package.read_bytes("Surfaces/alluvium-top.wkt").startswith(b"SRID=27700;")

    def test_index_in_subfolder(self, wkt_dir: Path, tmp_path: Path) -> None:
        path = zip_folder(wkt_dir, tmp_path / "nested.agsi", lambda name, data: (f"package/{name}", data))
        package = AgsiPackage.open(path)
        assert package.main_file == f"package/{MAIN_FILE}"
        _assert_mini(package)

    @pytest.mark.parametrize("folder", ["wkt_dir", "dxf_dir", "v1_dir"])
    def test_document_is_read_without_normalisation(self, folder: str, request: pytest.FixtureRequest) -> None:
        path: Path = request.getfixturevalue(folder)
        package = AgsiPackage.open(path)
        assert package.raw == json.loads((path / package.main_file).read_text(encoding="utf-8"))
        assert package.document.to_agsi() == package.raw


class TestSectionAlignment:
    """Section model -> alignmentID -> plan model's agsiModelAlignment, as in the AGSi guidance."""

    def test_linkage_and_coordinate_systems(self, wkt_dir: Path) -> None:
        package = AgsiPackage.open(wkt_dir)
        document = package.document
        section, plan = package.models
        assert section.alignment_id == "NB" and section.coord_system_id == "NB-Section-XZ"
        assert plan.agsi_model_element == [] and plan.coord_system_id == "BNG-XY"
        (alignment,) = plan.agsi_model_alignment
        assert alignment.alignment_id == section.alignment_id
        assert alignment.start_chainage == 0

        section_system = document.coordinate_system(section)
        plan_system = document.coordinate_system(plan)
        assert section_system is not None and section_system.system_id == "NB-Section-XZ"
        assert section_system.system_type == "XZ" and section_system.axis_name_x == "Chainage"
        assert plan_system is not None and plan_system.system_id == "BNG-XY"
        assert plan_system.system_type == "XY" and plan_system.global_xy_system == "EPSG:27700"

    @pytest.mark.parametrize("folder", ["wkt_dir", "dxf_dir"])
    def test_profile_chainage_is_station_along_alignment(self, folder: str, request: pytest.FixtureRequest) -> None:
        package = AgsiPackage.open(request.getfixturevalue(folder))
        geometry = package.models[1].agsi_model_alignment[0].agsi_geometry
        assert geometry is not None and geometry.geometry_type == "Line"
        alignment = package.read_geometry(geometry).single_line()
        assert alignment.shape == (6, 2)
        np.testing.assert_array_equal(alignment[0], [539251.3200000001, 179255.63])
        station = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(alignment, axis=0).T))])
        for line in _section_lines(package).values():
            np.testing.assert_allclose(line[:, 0], station, rtol=0, atol=1e-9)

    def test_wkt_and_dxf_profiles_are_identical(self, wkt_dir: Path, dxf_dir: Path) -> None:
        wkt_lines = _section_lines(AgsiPackage.open(wkt_dir))
        dxf_lines = _section_lines(AgsiPackage.open(dxf_dir))
        assert wkt_lines.keys() == dxf_lines.keys()
        for key, line in wkt_lines.items():
            np.testing.assert_array_equal(line, dxf_lines[key])


class TestVersions:
    @pytest.mark.parametrize("version", ["1.0", "1.0.0", "1.0.1", " 1.0.2 "])
    def test_supported(self, version: str) -> None:
        assert is_supported_version(version)

    @pytest.mark.parametrize("version", ["0.6", "1.1.0", "2.0.0", "1.0.1-beta", "v1.0.1", "", None, 1.0])
    def test_unsupported(self, version: Any) -> None:
        assert not is_supported_version(version)

    def test_v06_package_rejected(self, tmp_path: Path) -> None:
        path = _write_zip(
            tmp_path / "v06.agsi",
            {
                "index.json": {"agsiIndex": {"filePath": "main.json", "fileVersion": "0.6", "name": "Old"}},
                "main.json": {"agsiModel": [{"agsiModelElement": []}]},
            },
        )
        with pytest.raises(AGSIUnsupportedVersionError, match=r"'0\.6' \(index.json fileVersion\)"):
            AgsiPackage.open(path)

    @pytest.mark.parametrize("in_folder", [False, True])
    def test_document_without_ags_schema_rejected(self, tmp_path: Path, in_folder: bool) -> None:
        path = tmp_path / "main.json"
        path.write_text(json.dumps({"agsiModel": [{"modelName": "v0.6 style"}]}))
        with pytest.raises(AGSIUnsupportedVersionError, match="no 'agsSchema.version'"):
            AgsiPackage.open(tmp_path if in_folder else path)

    @pytest.mark.parametrize("version", ["0.6", "1.1.0", "2.0.0"])
    def test_unsupported_schema_version_rejected(self, tmp_path: Path, version: str) -> None:
        path = tmp_path / "main.json"
        path.write_text(json.dumps({"agsSchema": {"name": "AGSi", "version": version}, "agsiModel": []}))
        with pytest.raises(AGSIUnsupportedVersionError, match=f"'{version}' \\(agsSchema.version\\)"):
            AgsiPackage.open(path)

    def test_unsupported_index_version_rejected(self, tmp_path: Path) -> None:
        path = _write_zip(
            tmp_path / "x.agsi",
            {
                "index.json": {"agsiIndex": {"filePath": "main.json", "schemaVersion": "1.1.0"}},
                "main.json": {"agsSchema": V101_SCHEMA, "agsiModel": []},
            },
        )
        with pytest.raises(AGSIUnsupportedVersionError, match="index.json schemaVersion"):
            AgsiPackage.open(path)

    def test_index_version_mismatch_warns(self, tmp_path: Path) -> None:
        path = _write_zip(
            tmp_path / "x.agsi",
            {
                "index.json": {"agsiIndex": {"filePath": "main.json", "schemaVersion": "1.0.0"}},
                "main.json": {"agsSchema": V101_SCHEMA, "agsiModel": []},
            },
        )
        package = AgsiPackage.open(path)
        assert package.schema_version == "1.0.1"
        assert package.warnings == ["index.json schemaVersion '1.0.0' differs from agsSchema.version '1.0.1'."]

    def test_unsupported_version_is_invalid_data(self) -> None:
        assert issubclass(AGSIUnsupportedVersionError, AGSIInvalidDataError)


class TestFileResolution:
    def test_case_insensitive_match(self, wkt_dir: Path, tmp_path: Path) -> None:
        path = zip_folder(
            wkt_dir, tmp_path / "case.agsi", lambda name, data: (name.replace("Profiles/", "profiles/"), data)
        )
        package = AgsiPackage.open(path)
        lines = _section_lines(package)
        assert len(lines) == 3
        assert len(package.warnings) == 3 and all("case-insensitively" in w for w in package.warnings)

    def test_no_extension_substitution(self, dxf_dir: Path, tmp_path: Path) -> None:
        # A referenced file is never replaced by a file with another extension.
        path = zip_folder(dxf_dir, tmp_path / "renamed.agsi", lambda name, data: (name.replace(".dxf", ".txt"), data))
        package = AgsiPackage.open(path)
        with pytest.raises(AGSIDataFileIOError, match="not found"):
            package.read_bytes("Profiles/NB_Alignment-01-Surficial-Top.dxf")

    def test_percent_encoded_and_dot_segments(self, tmp_path: Path) -> None:
        path = _write_zip(
            tmp_path / "encoded.agsi",
            {
                "index.json": {"agsiIndex": {"filePath": "Model/main.json", "schemaVersion": "1.0.1"}},
                "Model/main.json": {"agsSchema": V101_SCHEMA, "agsiModel": []},
                "Model/Profiles/My Profile.txt": "LINESTRING (0 0, 1 1)",
                "Shared/line.txt": "LINESTRING (0 0, 2 2)",
            },
        )
        package = AgsiPackage.open(path)
        assert package.resolve_uri("Profiles/My%20Profile.txt") == "Model/Profiles/My Profile.txt"
        assert package.resolve_uri("./Profiles/My%20Profile.txt") == "Model/Profiles/My Profile.txt"
        assert package.resolve_uri("../Shared/line.txt") == "Shared/line.txt"
        assert package.resolve_uri("Shared/line.txt") == "Shared/line.txt"  # package-root fallback
        assert package.read_geometry("Shared/line.txt").lines[0][-1, 0] == 2

    @pytest.mark.parametrize(
        "uri",
        [
            "../outside.txt",
            "Profiles/../../outside.txt",
            "http://example.com/a.txt",
            "/abs.txt",
            "C:/a.txt",
            "file:///a.txt",
        ],
    )
    def test_rejects_uris_outside_package(self, wkt_zip: Path, uri: str) -> None:
        with pytest.raises(AGSIDataFileIOError):
            AgsiPackage.open(wkt_zip).resolve_uri(uri)

    def test_missing_file(self, wkt_dir: Path) -> None:
        package = AgsiPackage.open(wkt_dir)
        with pytest.raises(AGSIDataFileIOError, match="not found"):
            package.read_bytes("Profiles/does-not-exist.txt")


class TestErrors:
    def test_path_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(AGSIDataFileIOError, match="does not exist"):
            AgsiPackage.open(tmp_path / "missing.agsi")

    def test_zip_without_agsi_json(self, tmp_path: Path) -> None:
        with pytest.raises(AGSIDataFileIOError, match="No AGSi main JSON"):
            AgsiPackage.open(_write_zip(tmp_path / "x.agsi", {"other.json": {"a": 1}, "readme.txt": "hi"}))

    def test_folder_without_agsi_json(self, tmp_path: Path) -> None:
        (tmp_path / "other.json").write_text("{}")
        with pytest.raises(AGSIDataFileIOError, match="No AGSi main JSON"):
            AgsiPackage.open(tmp_path)

    def test_index_points_to_missing_file(self, tmp_path: Path) -> None:
        path = _write_zip(tmp_path / "x.agsi", {"index.json": {"agsiIndex": {"filePath": "main.json"}}})
        with pytest.raises(AGSIDataFileIOError, match="main.json"):
            AgsiPackage.open(path)

    def test_index_without_file_path(self, tmp_path: Path) -> None:
        with pytest.raises(AGSIInvalidDataError, match="filePath"):
            AgsiPackage.open(_write_zip(tmp_path / "x.agsi", {"index.json": {"agsiIndex": {}}}))

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "main.json"
        path.write_text("{not json")
        with pytest.raises(AGSIInvalidDataError, match="invalid JSON"):
            AgsiPackage.open(path)

    def test_not_agsi_json(self, tmp_path: Path) -> None:
        path = tmp_path / "main.json"
        path.write_text('{"something": "else"}')
        with pytest.raises(AGSIInvalidDataError, match="not an AGSi document"):
            AgsiPackage.open(path)

    def test_invalid_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "main.json"
        path.write_text(json.dumps({"agsSchema": V101_SCHEMA, "agsiModel": "nope"}))
        with pytest.raises(AGSIInvalidDataError, match="invalid AGSi document"):
            AgsiPackage.open(path)

    def test_utf8_bom(self, tmp_path: Path) -> None:
        path = tmp_path / "main.json"
        document = {"agsSchema": V101_SCHEMA, "agsiModel": [{"modelName": "BOM"}]}
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(document).encode())
        assert AgsiPackage.open(path).models[0].model_name == "BOM"
