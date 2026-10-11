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

import os
from typing import Optional, Union

from evo_schemas.components import Crs_V1_0_1
from evo_schemas.objects.geological_sections import GeologicalSections_V1_2_0

import evo.logging
from evo.data_converters.agsi.importer.agsi_reader import read_agsi_file
from evo.data_converters.agsi.importer.geological_sections import (
    build_geological_sections,
    check_crs,
    is_section_model,
    model_label,
)
from evo.data_converters.agsi.reader import AGSIUnsupportedGeometryError
from evo.objects.utils.data import ObjectDataClient

logger = evo.logging.getLogger("data_converters")


def get_geoscience_object_from_agsi(
    data_client: ObjectDataClient,
    filepath: str,
    coordinate_reference_system: Crs_V1_0_1,
    tags: Optional[dict[str, str]] = None,
    *,
    top_elevation: Optional[float] = None,
    base_elevation: Optional[float] = None,
) -> Union[GeologicalSections_V1_2_0, list[GeologicalSections_V1_2_0]]:
    """Build Evo ``geological-sections`` objects from the 2D section models of an AGSi 1.0.x package.

    Each model with ``agsiGeometryAreaFromLines`` elements becomes one object with one section.
    Other models (e.g. plan models that only hold alignments, or 3D models) are skipped.

    :param data_client: Client used to save the line geometry and attribute tables
        (``data_client.save_table``).
    :param filepath: Path to the ``.agsi`` package, extracted folder, or main JSON file.
    :param coordinate_reference_system: The CRS of the objects. The ``globalXYSystem`` declared in
        the AGSi file is not applied automatically.
    :param tags: (Optional) Additional tags to attach to the objects.
    :param top_elevation: (Optional) Top of units without a top profile; defaults to the model
        boundary ``topElevation``.
    :param base_elevation: (Optional) Base of units without a bottom profile; defaults to the model
        boundary ``bottomElevation``.
    :return: The object, or a list if the package has more than one section model.
    :raise AGSIDataFileIOError: If the package could not be read.
    :raise AGSIInvalidDataError: If the data is invalid or a required elevation is missing.
    :raise AGSIUnsupportedVersionError: If the package is not AGSi 1.0.x.
    :raise AGSIUnsupportedGeometryError: If there is no 2D section model, or its alignment is not
        straight or its profiles are not 2D.
    """
    package = read_agsi_file(filepath)

    object_tags = {
        "Source": f"{os.path.basename(filepath)} (via Evo Data Converters)",
        "Stage": "Experimental",
        "InputType": "AGSI",
    }
    if tags:
        object_tags.update(tags)

    models = [(index, model) for index, model in enumerate(package.models) if is_section_model(model)]
    if not models:
        raise AGSIUnsupportedGeometryError(
            f"'{os.path.basename(filepath)}' has no 2D section model (agsiGeometryAreaFromLines elements); "
            "only geological sections are supported."
        )
    for index, model in enumerate(package.models):
        if model.agsi_model_element and not is_section_model(model):
            logger.warning(f"Model '{model_label(model, index)}' has no section areas; skipped.")

    objects = []
    for index, model in models:
        name = model_label(model, index)
        check_crs(package, model, name, coordinate_reference_system)
        objects.append(
            build_geological_sections(
                package,
                model,
                data_client,
                coordinate_reference_system,
                name=name,
                tags=object_tags,
                top_elevation=top_elevation,
                base_elevation=base_elevation,
            )
        )
    return objects[0] if len(objects) == 1 else objects
