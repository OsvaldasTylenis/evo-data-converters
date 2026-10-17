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

"""Map AGSi 2D section models to Evo ``geological-sections`` objects."""

import math
import re
from typing import Any, Optional

import numpy as np
import pyarrow as pa
from evo_schemas.components import (
    BoundingBox_V1_0_1,
    ContinuousAttribute_V1_1_0,
    Crs_V1_0_1,
    Crs_V1_0_1_EpsgCode,
    EmbeddedLineGeometry_V1_0_0,
    EmbeddedLineGeometry_V1_0_0_Chunks,
    EmbeddedLineGeometry_V1_0_0_Vertices,
    Material_V1_0_1,
    NanContinuous_V1_0_1,
    OneOfAttribute_V1_2_0_Item,
    StringAttribute_V1_1_0,
)
from evo_schemas.elements import FloatArray1_V1_0_1, ReversibleIndex_V1_0_0, StringArray_V1_0_1
from evo_schemas.objects.geological_sections import (
    GeologicalSections_V1_2_0,
    GeologicalSections_V1_2_0_Folder,
    GeologicalSections_V1_2_0_Folder_Items_SectionIndex,
    GeologicalSections_V1_2_0_GmEmbeddedPolygonVolume,
    GeologicalSections_V1_2_0_GmEmbeddedPolylineSurface,
    GeologicalSections_V1_2_0_Section,
    GeologicalSections_V1_2_0_Section_IndicesRange,
)

import evo.logging
from evo.objects.utils.data import ObjectDataClient

from ..geometry import ModelSection, SectionLine, build_model_section, find_model_alignment
from ..reader import AGSIInvalidDataError, AgsiPackage
from ..reader.models import (
    AgsiGeometryAreaFromLines,
    AgsiModel,
    AgsiModelAlignment,
    AgsiModelElement,
    AgsProjectCoordinateSystem,
)

logger = evo.logging.getLogger("data_converters")

VOLUME_FEATURE = "OutputVolume"
CONTACT_FEATURE = "ContactSurface"
BOUNDARY_FEATURE = "BoundarySurface"

# Material colours (0xRRGGBB) for elements without colourRGB.
PALETTE = (
    0x1F77B4,
    0xFF7F0E,
    0x2CA02C,
    0xD62728,
    0x9467BD,
    0x8C564B,
    0xE377C2,
    0x7F7F7F,
    0xBCBD22,
    0x17BECF,
)

_HEX_COLOUR = re.compile(r"#?([0-9A-Fa-f]{6})")


def is_section_model(model: AgsiModel) -> bool:
    """Whether ``model`` has ``agsiGeometryAreaFromLines`` elements, i.e. is a 2D section model."""
    return any(isinstance(element.agsi_geometry, AgsiGeometryAreaFromLines) for element in model.agsi_model_element)


def model_label(model: AgsiModel, index: int) -> str:
    return model.model_name or model.model_id or f"Model {index + 1}"


def element_label(element: AgsiModelElement, key: str) -> str:
    return element.element_name or element.element_id or key


def evo_colour(colour_rgb: Optional[str], index: int) -> int:
    """Evo material colour (alpha, blue, green, red from the highest to the lowest byte).

    Uses AGSi ``colourRGB`` (``#RRGGBB``) when valid, else a fixed palette entry.
    """
    match = _HEX_COLOUR.fullmatch(colour_rgb.strip()) if colour_rgb else None
    if colour_rgb and match is None:
        logger.warning(f"colourRGB '{colour_rgb}' is not a #RRGGBB colour; a default colour is used.")
    rgb = int(match.group(1), 16) if match else PALETTE[index % len(PALETTE)]
    red, green, blue = (rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF
    return (0xFF << 24) | (blue << 16) | (green << 8) | red


def build_geological_sections(
    package: AgsiPackage,
    model: AgsiModel,
    data_client: ObjectDataClient,
    coordinate_reference_system: Crs_V1_0_1,
    *,
    name: str,
    tags: Optional[dict[str, str]] = None,
    top_elevation: Optional[float] = None,
    base_elevation: Optional[float] = None,
) -> GeologicalSections_V1_2_0:
    """Build one ``geological-sections`` object holding the single section of a 2D section model.

    The section is placed on the model's straight alignment (see
    :func:`~evo.data_converters.agsi.geometry.build_model_section`). Section coordinates are
    (u, v) = (chainage, elevation); the section origin is the world point at chainage 0 and
    elevation 0.

    :raise AGSIInvalidDataError: If the model has no section polygons, or its data is invalid.
    :raise AGSIUnsupportedGeometryError: If its alignment or profiles are not supported.
    """
    alignment = find_model_alignment(package.document, model)
    if alignment is None:
        logger.warning(
            f"Model '{name}' has no alignmentID; the section is placed in a local frame with chainage "
            "along +X from (0, 0), not at real-world coordinates."
        )
    section = build_model_section(package, model, top_elevation=top_elevation, base_elevation=base_elevation)
    geometry = section.geometry
    if not geometry.volumes:
        raise AGSIInvalidDataError(f"Model '{name}' has no section areas with a positive thickness.")

    vertices = data_client.save_table(
        pa.table(
            {
                "u": pa.array(geometry.vertices[:, 0], type=pa.float64()),
                "v": pa.array(geometry.vertices[:, 1], type=pa.float64()),
            }
        )
    )
    chunks = data_client.save_table(
        pa.table(
            {
                "offset": pa.array(geometry.chunks[:, 0], type=pa.uint64()),
                "count": pa.array(geometry.chunks[:, 1], type=pa.uint64()),
            }
        )
    )
    line_geometry = EmbeddedLineGeometry_V1_0_0(
        vertices=EmbeddedLineGeometry_V1_0_0_Vertices(**vertices),
        chunks=EmbeddedLineGeometry_V1_0_0_Chunks(**chunks),
    )

    labels = {key: element_label(element, key) for key, element in section.elements.items()}
    volumes = [
        GeologicalSections_V1_2_0_GmEmbeddedPolygonVolume(
            name=labels[volume.key],
            parts=[ReversibleIndex_V1_0_0(index=int(chunk), reversed=bool(rev)) for chunk, rev in volume.parts],
            description=section.elements[volume.key].description,
            feature=VOLUME_FEATURE,
            material_key=volume.key,
            layer=volume.key,
        )
        for volume in geometry.volumes
    ]
    surfaces = [_surface(line, section, labels) for line in geometry.lines]
    materials = [
        Material_V1_0_1(key=key, name=labels[key], color=evo_colour(element.colour_rgb, index))
        for index, (key, element) in enumerate(section.elements.items())
    ]

    world = section.frame.section_to_world(geometry.vertices)
    low, high = world.min(axis=0), world.max(axis=0)
    bounding_box = BoundingBox_V1_0_1(
        min_x=float(low[0]),
        max_x=float(high[0]),
        min_y=float(low[1]),
        max_y=float(high[1]),
        min_z=float(low[2]),
        max_z=float(high[2]),
    )

    evo_section = GeologicalSections_V1_2_0_Section(
        name=name,
        origin=[float(value) for value in section.frame.origin],
        rotation=section.frame.rotation,
        description=model.description,
        volumes=GeologicalSections_V1_2_0_Section_IndicesRange(offset=0, count=len(volumes)),
        surfaces=GeologicalSections_V1_2_0_Section_IndicesRange(offset=0, count=len(surfaces)) if surfaces else None,
    )
    folder_name = (alignment.alignment_name or alignment.alignment_id or name) if alignment else name

    return GeologicalSections_V1_2_0(
        name=name,
        uuid=None,
        description=model.description,
        tags=tags,
        extensions={"agsi": _agsi_extension(package, model, alignment, section, surfaces)},
        bounding_box=bounding_box,
        coordinate_reference_system=coordinate_reference_system,
        folders=[
            GeologicalSections_V1_2_0_Folder(
                name=folder_name, items=[GeologicalSections_V1_2_0_Folder_Items_SectionIndex(section_index=0)]
            )
        ],
        line_geometry=line_geometry,
        sections=[evo_section],
        volumes=volumes,
        surfaces=surfaces,
        materials=materials,
        volume_attributes=_volume_attributes(data_client, section, [volume.key for volume in geometry.volumes]),
        layer_order=list(section.elements),
    )


def _surface(
    line: SectionLine, section: ModelSection, labels: dict[str, str]
) -> GeologicalSections_V1_2_0_GmEmbeddedPolylineSurface:
    """Surface for a top/bottom line: ``ContactSurface``, or ``BoundarySurface`` if generated."""
    name = " / ".join(
        [f"Base of {labels[key]}" for key in line.bottom_of] + [f"Top of {labels[key]}" for key in line.top_of]
    )
    descriptions: list[str] = []
    for key in line.bottom_of + line.top_of:
        geometry = section.elements[key].agsi_geometry
        assert isinstance(geometry, AgsiGeometryAreaFromLines)
        source = geometry.agsi_geometry_bottom if key in line.bottom_of else geometry.agsi_geometry_top
        if source is not None and source.description and source.description not in descriptions:
            descriptions.append(source.description)
    if line.generated:
        elevation = float(section.geometry.line_vertices(line)[0, 1])
        missing = "bottom" if line.bottom_of else "top"
        descriptions.append(
            f"Horizontal line at elevation {elevation} generated for units without a {missing} profile."
        )
    return GeologicalSections_V1_2_0_GmEmbeddedPolylineSurface(
        name=name,
        parts=[ReversibleIndex_V1_0_0(index=int(chunk), reversed=False) for chunk in line.parts],
        description="; ".join(descriptions) or None,
        feature=BOUNDARY_FEATURE if line.generated else CONTACT_FEATURE,
    )


def _volume_attributes(
    data_client: ObjectDataClient, section: ModelSection, keys: list[str]
) -> list[OneOfAttribute_V1_2_0_Item]:
    """One row per volume: element ID and type, then one attribute per parameter code (and case).

    Parameters with numeric values only become continuous attributes; others become string
    attributes. ``valueProfile`` parameters are not converted.
    """
    columns: dict[str, dict[str, Any]] = {}
    for key, element in section.elements.items():
        for parameter in element.agsi_data_parameter_value:
            if parameter.code_id is None:
                logger.warning(f"Element '{key}': agsiDataParameterValue without codeID skipped.")
                continue
            value: Any = parameter.value_numeric if parameter.value_numeric is not None else parameter.value_text
            if value is None:
                if parameter.value_profile is not None:
                    logger.warning(f"Element '{key}': valueProfile of '{parameter.code_id}' is not converted.")
                continue
            column = _parameter_column(parameter.code_id, parameter.case_id)
            values = columns.setdefault(column, {})
            if key in values:
                logger.warning(f"Element '{key}': repeated parameter '{column}'; the first value is kept.")
                continue
            values[key] = value

    attributes: list[OneOfAttribute_V1_2_0_Item] = [
        _string_attribute(data_client, "elementID", [section.elements[key].element_id for key in keys]),
        _string_attribute(data_client, "elementType", [section.elements[key].element_type for key in keys]),
    ]
    for column, values in columns.items():
        row_values = [values.get(key) for key in keys]
        if all(isinstance(value, float) or value is None for value in values.values()):
            attributes.append(_continuous_attribute(data_client, column, row_values))
        else:
            attributes.append(
                _string_attribute(data_client, column, [None if v is None else _text(v) for v in row_values])
            )
    return attributes


def _parameter_column(code_id: str, case_id: Optional[str]) -> str:
    return code_id if case_id is None else f"{code_id} ({case_id})"


def _parameter_columns(section: ModelSection) -> dict[str, dict[str, str]]:
    """codeID and caseID of the attribute columns of parameters with a caseID."""
    columns: dict[str, dict[str, str]] = {}
    for element in section.elements.values():
        for parameter in element.agsi_data_parameter_value:
            if parameter.code_id is not None and parameter.case_id is not None:
                column = _parameter_column(parameter.code_id, parameter.case_id)
                columns.setdefault(column, {"codeID": parameter.code_id, "caseID": parameter.case_id})
    return columns


def _text(value: Any) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _string_attribute(data_client: ObjectDataClient, name: str, values: list[Optional[str]]) -> StringAttribute_V1_1_0:
    table = data_client.save_table(pa.table({"values": pa.array(values, type=pa.string())}))
    return StringAttribute_V1_1_0(name=name, key=name, values=StringArray_V1_0_1(**table))


def _continuous_attribute(
    data_client: ObjectDataClient, name: str, values: list[Optional[float]]
) -> ContinuousAttribute_V1_1_0:
    array = np.array([math.nan if value is None else value for value in values], dtype=np.float64)
    table = data_client.save_table(pa.table({"values": pa.array(array, type=pa.float64())}))
    return ContinuousAttribute_V1_1_0(
        name=name, key=name, nan_description=NanContinuous_V1_0_1(values=[]), values=FloatArray1_V1_0_1(**table)
    )


def _agsi_extension(
    package: AgsiPackage,
    model: AgsiModel,
    alignment: Optional[AgsiModelAlignment],
    section: ModelSection,
    surfaces: list[GeologicalSections_V1_2_0_GmEmbeddedPolylineSurface],
) -> dict[str, Any]:
    """AGSi identifiers of the source model, kept under ``extensions.agsi``.

    ``surfaces`` records, for each Evo surface in order, the volume layers (element keys) whose
    top (``topOf``) and bottom (``bottomOf``) it is. An element that pinches out has several
    volumes; this contact provenance lets the exporter write it as one AGSi element again with
    its complete top and bottom profiles.
    """
    extension: dict[str, Any] = {
        "schemaVersion": package.schema_version,
        "modelID": model.model_id,
        "modelName": model.model_name,
        "coordSystemID": model.coord_system_id,
        "alignmentID": model.alignment_id,
    }
    if alignment is not None:
        extension["alignmentName"] = alignment.alignment_name
        extension["startChainage"] = section.frame.start_chainage
        extension["alignmentLength"] = section.frame.length
    parameter_columns = _parameter_columns(section)
    if parameter_columns:
        extension["parameterColumns"] = parameter_columns
    extension["surfaces"] = [
        {"name": surface.name, "topOf": list(line.top_of), "bottomOf": list(line.bottom_of)}
        for surface, line in zip(surfaces, section.geometry.lines)
    ]
    return {key: value for key, value in extension.items() if value is not None}


def plan_coordinate_system(package: AgsiPackage, model: AgsiModel) -> Optional[AgsProjectCoordinateSystem]:
    """Coordinate system of the model holding the alignment of ``model`` (its world XY system)."""
    alignment = find_model_alignment(package.document, model)
    if alignment is None:
        return None
    owner = next(
        candidate
        for candidate in package.document.agsi_model
        if any(item is alignment for item in candidate.agsi_model_alignment)
    )
    if not owner.coord_system_id:
        return None
    return package.document.coordinate_system(owner)


def check_crs(package: AgsiPackage, model: AgsiModel, name: str, crs: Crs_V1_0_1) -> None:
    """Log how the requested CRS relates to the coordinate system declared in the AGSi file.

    The declared ``globalXYSystem`` is never applied automatically: it may be inferred rather than
    stated by the data producer, so the CRS must be given by the caller.
    """
    system = plan_coordinate_system(package, model)
    if system is None:
        return
    shifts = (system.transform_shift_x, system.transform_shift_y, system.transform_xy_rotation)
    if any(value not in (None, 0.0) for value in shifts) or system.transform_xy_scale_factor not in (None, 1.0):
        logger.warning(
            f"Model '{name}': coordinate system '{system.system_id}' defines a transformation to its global "
            "system; it is not applied, coordinates are used as stored."
        )
    declared = system.global_xy_system
    if declared is None:
        return
    if crs == "unspecified":
        logger.warning(
            f"Model '{name}': no coordinate reference system was given. The AGSi coordinate system "
            f"'{system.system_id}' declares globalXYSystem '{declared}'; pass coordinate_reference_system "
            "to use it. It is not applied automatically."
        )
        return
    match = re.fullmatch(r"\s*EPSG:\s*(\d+)\s*", declared, re.IGNORECASE)
    if isinstance(crs, Crs_V1_0_1_EpsgCode) and match and int(match.group(1)) != crs.epsg_code:
        logger.warning(
            f"Model '{name}': coordinate_reference_system EPSG:{crs.epsg_code} differs from the AGSi "
            f"globalXYSystem '{declared}'."
        )
