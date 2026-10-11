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

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import UUID

import nest_asyncio
from evo_schemas import schema_lookup

import evo.logging
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.objects.client import ObjectAPIClient
from evo.objects.data import ObjectSchema
from evo.objects.utils.data import ObjectDataClient

from .errors import AGSIExporterException, UnsupportedObjectError, UnsupportedSectionGeometryError
from .geological_sections import GEOLOGICAL_SECTIONS_CLASSES, SectionsSource, build_agsi_export

logger = evo.logging.getLogger("data_converters")

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget

__all__ = [
    "AGSIExporterException",
    "UnsupportedObjectError",
    "UnsupportedSectionGeometryError",
    "export_agsi",
    "export_blocksync_agsi",
]


def _download_source(
    service_client: ObjectAPIClient, data_client: ObjectDataClient, object_metadata: EvoObjectMetadata
) -> SectionsSource:
    object_id = object_metadata.object_id
    version_id = object_metadata.version_id
    downloaded = asyncio.run(service_client.download_object_by_id(object_id, version_id))
    geoscience_object_dict: dict[str, Any] = downloaded.as_dict()

    schema = ObjectSchema.from_id(geoscience_object_dict["schema"])
    object_class = schema_lookup.get(str(schema))
    if not object_class:
        raise UnsupportedObjectError(f"Unknown Geoscience Object schema '{schema}'")
    geoscience_object = object_class.from_dict(geoscience_object_dict)
    if not isinstance(geoscience_object, GEOLOGICAL_SECTIONS_CLASSES):
        raise UnsupportedObjectError(
            f"Exporting {geoscience_object.__class__.__name__} Geoscience Objects to AGSi is not supported; "
            "only geological-sections objects can be exported."
        )

    def load_table(component: Any) -> Any:
        return asyncio.run(data_client.download_table(object_id, version_id, component.as_dict()))

    return SectionsSource(
        geoscience_object=geoscience_object,
        load_table=load_table,
        object_id=str(object_id),
        version_id=version_id,
    )


def export_agsi(
    filepath: Union[str, Path],
    objects: list[EvoObjectMetadata],
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> None:
    """Export Evo ``geological-sections`` objects to an AGSi 1.0.1 package (``.agsi`` ZIP file).

    Every section becomes an AGSi section model of ``agsiGeometryAreaFromLines`` elements (one per
    Evo volume) with WKT top and base profiles, linked to a straight alignment in a plan model.
    The package is built completely in memory and written only if every object converts.

    :param filepath: Path of the ``.agsi`` file to create.
    :param objects: List of EvoObjectMetadata objects containing the UUID and version of the Evo objects to export.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.

    One of evo_workspace_metadata or service_manager_widget is required.

    :raise UnsupportedObjectError: If the type of a requested object is not supported.
    :raise UnsupportedSectionGeometryError: If a section is not a vertical plane or a volume cannot
        be written as one top and one bottom profile.
    :raise AGSIExporterException: If there is nothing to export or the object data is invalid.
    :raise MissingConnectionDetailsError: If no connection details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    """
    if not objects:
        raise AGSIExporterException("No objects to export.")
    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata=evo_workspace_metadata, service_manager_widget=service_manager_widget
    )
    nest_asyncio.apply()

    sources = [_download_source(service_client, data_client, object_metadata) for object_metadata in objects]
    export = build_agsi_export(sources, title=Path(filepath).stem)
    export.write(filepath)


def export_blocksync_agsi(
    filepath: str,
    object_id: UUID,
    version_id: Optional[str] = None,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> None:
    """Export a BlockSync block model to a AGSI file.

    :param filepath: Path of the AGSI file to create.
    :param object_id: UUID of the BlockSync block model object to export.
    :param version_id: (Optional) Version of the object to export. Defaults to the latest version.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.

    One of evo_workspace_metadata or service_manager_widget is required.

    :raise UnsupportedObjectError: If the block model type is not supported.
    :raise MissingConnectionDetailsError: If no connection details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    """
    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata=evo_workspace_metadata, service_manager_widget=service_manager_widget
    )

    # TODO: Implement exporting a BlockSync block model to a AGSI file.
    #   1. Download the block model identified by ``object_id`` / ``version_id``.
    #   2. Convert it into your AGSI representation (raise UnsupportedObjectError
    #      for block model types you do not support).
    #   3. Write the converted data out to ``filepath``.
    raise NotImplementedError("Implement export_blocksync_agsi for your AGSI file format.")
