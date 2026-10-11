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

from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pyproj import CRS
from evo_schemas.components import BaseSpatialDataProperties_V1_0_1, Crs_V1_0_1_EpsgCode, Crs_V1_0_1_OgcWkt
from evo_schemas.objects.geological_sections import GeologicalSections_V1_2_0

from conftest import edit_document
from evo.common.exceptions import NotFoundException
from evo.data_converters.common import EvoWorkspaceMetadata
from evo.data_converters.common.exceptions import ConflictingConnectionDetailsError, MissingConnectionDetailsError
from evo.data_converters.agsi.importer.agsi_to_evo import convert_agsi
from evo.data_converters.agsi.reader import AGSIInvalidDataError, AGSIUnsupportedGeometryError
from evo.objects.data import ObjectMetadata


_WKT2_EXAMPLE = """\
GEOGCRS["WGS 84",
    DATUM["World Geodetic System 1984",
        ELLIPSOID["WGS 84", 6378137, 298.257223563,
            LENGTHUNIT["metre", 1]]],
    PRIMEM["Greenwich", 0,
        ANGLEUNIT["degree", 0.0174532925199433]],
    CS[ellipsoidal, 2],
        AXIS["geodetic latitude", north,
            ORDER[1],
            ANGLEUNIT["degree", 0.0174532925199433]],
        AXIS["geodetic longitude", east,
            ORDER[2],
            ANGLEUNIT["degree", 0.0174532925199433]],
    ID["EPSG", 4326]]"""


def test_convert_agsi_success() -> None:
    filepath = "dummy_file.txt"
    coordinate_reference_system = _WKT2_EXAMPLE
    evo_workspace_metadata = EvoWorkspaceMetadata(hub_url="http://example.com")
    tags = {"tag1": "value1"}
    upload_path = "upload/path"

    mock_geoscience_object = MagicMock(spec=BaseSpatialDataProperties_V1_0_1)
    mock_metadata = MagicMock(spec=ObjectMetadata)

    with (
        patch(
            "evo.data_converters.agsi.importer.agsi_to_evo.create_evo_object_service_and_data_client"
        ) as mock_create_client,
        patch("evo.data_converters.agsi.importer.agsi_to_evo.publish_geoscience_objects_sync") as mock_publish,
        patch(
            "evo.data_converters.agsi.importer.utils.get_geoscience_object_from_agsi",
            return_value=mock_geoscience_object,
        ) as mock_get_geoscience_object,
    ):
        mock_create_client.return_value = (MagicMock(), MagicMock())
        mock_publish.return_value = [mock_metadata]

        result = convert_agsi(
            filepath,
            evo_workspace_metadata=evo_workspace_metadata,
            tags=tags,
            upload_path=upload_path,
            publish_objects=True,
            coordinate_reference_system=coordinate_reference_system,
        )

        assert result == [mock_metadata]
        mock_get_geoscience_object.assert_called_once()
        assert mock_get_geoscience_object.call_args[0][1] == filepath
        mock_publish.assert_called_once_with(
            [mock_geoscience_object],
            mock_create_client.return_value[0],
            mock_create_client.return_value[1],
            upload_path,
            False,
        )


def test_convert_agsi_no_publish() -> None:
    filepath = "dummy_file.txt"
    evo_workspace_metadata = EvoWorkspaceMetadata()
    tags = {"tag1": "value1"}
    upload_path = "upload/path"

    mock_geoscience_object = MagicMock(spec=BaseSpatialDataProperties_V1_0_1)

    with (
        patch(
            "evo.data_converters.agsi.importer.agsi_to_evo.create_evo_object_service_and_data_client"
        ) as mock_create_client,
        patch(
            "evo.data_converters.agsi.importer.utils.get_geoscience_object_from_agsi",
            return_value=mock_geoscience_object,
        ),
    ):
        mock_create_client.return_value = (MagicMock(), MagicMock())

        result = convert_agsi(
            filepath,
            evo_workspace_metadata=evo_workspace_metadata,
            tags=tags,
            upload_path=upload_path,
            publish_objects=False,
        )

    assert result == [mock_geoscience_object]


def test_convert_agsi_missing_connection_details_error() -> None:
    filepath = "dummy_file.txt"

    with pytest.raises(MissingConnectionDetailsError):
        convert_agsi(filepath, publish_objects=False)


def test_convert_agsi_conflicting_connection_details_error() -> None:
    filepath = "dummy_file.txt"
    evo_workspace_metadata = EvoWorkspaceMetadata()
    service_manager_widget = MagicMock()

    with pytest.raises(ConflictingConnectionDetailsError):
        convert_agsi(
            filepath,
            evo_workspace_metadata=evo_workspace_metadata,
            service_manager_widget=service_manager_widget,
            publish_objects=False,
        )


@pytest.mark.parametrize(
    "input_crs, expected_crs",
    [
        (4326, Crs_V1_0_1_EpsgCode(epsg_code=4326)),
        ("EPSG:4326", Crs_V1_0_1_EpsgCode(epsg_code=4326)),
        (_WKT2_EXAMPLE, Crs_V1_0_1_OgcWkt(ogc_wkt=CRS.from_wkt(_WKT2_EXAMPLE).to_wkt("WKT2_2019"))),
        (None, "unspecified"),
    ],
)
def test_coordinate_reference_system(input_crs: Any, expected_crs: Any) -> None:
    filepath = "dummy_file.txt"
    evo_workspace_metadata = EvoWorkspaceMetadata(hub_url="http://example.com")

    mock_geoscience_object = MagicMock(spec=BaseSpatialDataProperties_V1_0_1)

    with (
        patch(
            "evo.data_converters.agsi.importer.agsi_to_evo.create_evo_object_service_and_data_client"
        ) as mock_create_client,
        patch(
            "evo.data_converters.agsi.importer.utils.get_geoscience_object_from_agsi",
            return_value=mock_geoscience_object,
        ) as mock_get_obj,
    ):
        mock_create_client.return_value = (MagicMock(), MagicMock())

        convert_agsi(
            filepath,
            evo_workspace_metadata=evo_workspace_metadata,
            coordinate_reference_system=input_crs,
            publish_objects=False,
        )

        crs_arg = mock_get_obj.call_args[0][2]
        assert crs_arg == expected_crs


def test_coordinate_reference_system_conflicts_with_epsg_code() -> None:
    filepath = "dummy_file.txt"
    evo_workspace_metadata = EvoWorkspaceMetadata(hub_url="http://example.com")

    with pytest.raises(ValueError, match="Both epsg_code and coordinate_reference_system were provided"):
        convert_agsi(
            filepath,
            4326,
            evo_workspace_metadata,
            coordinate_reference_system=4326,
            publish_objects=False,
        )


_CLIENTS = "evo.data_converters.agsi.importer.agsi_to_evo.create_evo_object_service_and_data_client"
_PUBLISH = "evo.data_converters.agsi.importer.agsi_to_evo.publish_geoscience_objects_sync"


@pytest.mark.usefixtures("block_network")
class TestConvertStraightSection:
    """``convert_agsi`` with the real object builder; Evo clients and publishing are mocked (offline)."""

    def test_publish_arguments(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        object_client = MagicMock()
        metadata = MagicMock(spec=ObjectMetadata)
        with (
            patch(_CLIENTS, return_value=(object_client, local_data_client)) as create_clients,
            patch(_PUBLISH, return_value=[metadata]) as publish,
        ):
            result = convert_agsi(
                str(straight_wkt_dir),
                evo_workspace_metadata=EvoWorkspaceMetadata(hub_url="http://example.com"),
                tags={"Project": "Silvertown"},
                upload_path="agsi/silvertown",
                overwrite_existing_objects=True,
                coordinate_reference_system="EPSG:27700",
                base_elevation=-50.0,
            )

        assert result == [metadata]
        create_clients.assert_called_once()
        publish.assert_called_once()
        objects, client, data_client, path, overwrite = publish.call_args.args
        assert (client, data_client, path, overwrite) == (object_client, local_data_client, "agsi/silvertown", True)
        (obj,) = objects
        assert isinstance(obj, GeologicalSections_V1_2_0)
        assert obj.name == "Silvertown mini NB section"
        assert obj.coordinate_reference_system == Crs_V1_0_1_EpsgCode(epsg_code=27700)
        assert obj.tags is not None and obj.tags["Project"] == "Silvertown"
        assert len(obj.volumes) == 2 and len(obj.surfaces) == 3
        # Referenced data is only saved to the local cache; nothing is uploaded or sent to a service.
        assert local_data_client.save_table.called
        assert not local_data_client.upload_referenced_data.called
        assert not object_client.method_calls

    def test_no_publish_returns_objects(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        with patch(_CLIENTS, return_value=(MagicMock(), local_data_client)), patch(_PUBLISH) as publish:
            result = convert_agsi(
                str(straight_wkt_dir),
                evo_workspace_metadata=EvoWorkspaceMetadata(),
                publish_objects=False,
                coordinate_reference_system=27700,
                base_elevation=-50.0,
            )
        publish.assert_not_called()
        (obj,) = result
        assert isinstance(obj, GeologicalSections_V1_2_0)
        assert obj.coordinate_reference_system == Crs_V1_0_1_EpsgCode(epsg_code=27700)

    def test_multiple_section_models_are_flattened(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        def duplicate(document: dict[str, Any]) -> None:
            second = dict(document["agsiModel"][0], modelID="Second", modelName="Second section")
            document["agsiModel"].append(second)

        edit_document(straight_wkt_dir, duplicate)
        with patch(_CLIENTS, return_value=(MagicMock(), local_data_client)), patch(_PUBLISH) as publish:
            publish.side_effect = lambda objects, *args: [MagicMock(spec=ObjectMetadata) for _ in objects]
            result = convert_agsi(
                str(straight_wkt_dir),
                evo_workspace_metadata=EvoWorkspaceMetadata(hub_url="http://example.com"),
                base_elevation=-50.0,
            )
        objects = publish.call_args.args[0]
        assert [obj.name for obj in objects] == ["Silvertown mini NB section", "Second section"]
        assert len(result) == 2

    @pytest.mark.parametrize(
        "folder, kwargs, error, match",
        [
            ("wkt_dir", {"base_elevation": -50.0}, AGSIUnsupportedGeometryError, "not straight"),
            ("straight_wkt_dir", {}, AGSIInvalidDataError, "base elevation is required"),
        ],
    )
    def test_errors_raised_before_publishing(
        self,
        local_data_client: MagicMock,
        request: pytest.FixtureRequest,
        folder: str,
        kwargs: dict[str, Any],
        error: type[Exception],
        match: str,
    ) -> None:
        with patch(_CLIENTS, return_value=(MagicMock(), local_data_client)), patch(_PUBLISH) as publish:
            with pytest.raises(error, match=match):
                convert_agsi(
                    str(request.getfixturevalue(folder)),
                    evo_workspace_metadata=EvoWorkspaceMetadata(hub_url="http://example.com"),
                    coordinate_reference_system=27700,
                    **kwargs,
                )
        publish.assert_not_called()

    def test_publish_flow_with_fake_clients(self, local_data_client: MagicMock, straight_wkt_dir: Path) -> None:
        """Run the shared publishing code against in-memory async clients."""
        metadata = MagicMock(spec=ObjectMetadata)
        object_client = MagicMock()
        object_client.download_object_by_path = AsyncMock(side_effect=NotFoundException(404, "Not found", None, None))
        object_client.create_geoscience_object = AsyncMock(return_value=metadata)
        object_client.update_geoscience_object = AsyncMock()
        local_data_client.upload_referenced_data = AsyncMock()

        with patch(_CLIENTS, return_value=(object_client, local_data_client)):
            result = convert_agsi(
                str(straight_wkt_dir),
                evo_workspace_metadata=EvoWorkspaceMetadata(hub_url="http://example.com"),
                upload_path="agsi",
                coordinate_reference_system=27700,
                base_elevation=-50.0,
            )

        assert result == [metadata]
        path = "agsi/Silvertown mini NB section.json"
        object_client.download_object_by_path.assert_awaited_once_with(path)
        (create_path, document), _ = object_client.create_geoscience_object.await_args
        assert create_path == path
        local_data_client.upload_referenced_data.assert_awaited_once_with(document)
        object_client.update_geoscience_object.assert_not_awaited()
        assert document["schema"] == "/objects/geological-sections/1.2.0/geological-sections.schema.json"
        assert document["coordinate_reference_system"] == {"epsg_code": 27700}
        assert len(document["sections"]) == 1 and len(document["volumes"]) == 2
        # Every referenced table was written to the local cache first.
        cached = {p.name for p in local_data_client.cache_location.iterdir()}
        assert {document["line_geometry"]["vertices"]["data"], document["line_geometry"]["chunks"]["data"]} <= cached
