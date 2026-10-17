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

"""Map Evo ``geological-sections`` objects to AGSi 1.0.1 section models.

Each Evo section becomes an AGSi model of ``agsiGeometryAreaFromLines`` elements, linked by
``alignmentID`` to a straight two-point ``agsiModelAlignment`` held in a plan (XY) model. Section
coordinates are written unchanged: chainage = u and elevation = v plus the elevation of the
section origin.

An AGSi element whose unit pinches out is several Evo volumes (a polygon cannot be split into
disconnected areas or touch itself). Objects created by the importer record which surface is the
top and bottom of each element (``extensions.agsi.surfaces``); with that provenance each element
is written once with its complete top and bottom lines, after checking that its volumes are
exactly the polygons between them. Without it, volumes with the same element ID are merged only
where they meet exactly at pinch-out vertices, else written as ``"(part n)"`` elements.

A volume polygon is written as a top and a bottom profile only when that split is unique: see
:func:`ring_to_profiles`. Anything else is rejected rather than approximated.
"""

import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from evo_schemas.components import (
    CategoryAttribute_V1_1_0,
    ContinuousAttribute_V1_1_0,
    Crs_V1_0_1_EpsgCode,
    Crs_V1_0_1_OgcWkt,
    IntegerAttribute_V1_1_0,
    StringAttribute_V1_1_0,
)
from evo_schemas.objects.geological_sections import (
    GeologicalSections_V1_0_0,
    GeologicalSections_V1_1_0,
    GeologicalSections_V1_2_0,
)
from pyproj import CRS
from pyproj.exceptions import CRSError
from scipy.spatial.transform import Rotation

import evo.logging

from ..geometry import SectionUnit, build_section_geometry, profile_to_section
from ..reader import AGSIInvalidDataError
from .errors import AGSIExporterException, UnsupportedSectionGeometryError

logger = evo.logging.getLogger("data_converters")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

GeologicalSectionsObject = Union[GeologicalSections_V1_0_0, GeologicalSections_V1_1_0, GeologicalSections_V1_2_0]
GEOLOGICAL_SECTIONS_CLASSES = (GeologicalSections_V1_0_0, GeologicalSections_V1_1_0, GeologicalSections_V1_2_0)

# Loads the table of a data reference (e.g. ``line_geometry.vertices``) of the object.
TableLoader = Callable[[Any], pa.Table]

AGSI_VERSION = "1.0.1"
AGSI_LINK = "https://ags-data-format-wg.gitlab.io/agsi/agsi_standard/1.0.1/"
PRODUCED_BY = "Evo Data Converters (evo-data-converters-agsi)"
INDEX_FILE_NAME = "index.json"
BOUNDARY_FEATURE = "BoundarySurface"
# Volume attributes written by the importer that hold element fields rather than parameters.
ELEMENT_ID_ATTRIBUTE = "elementID"
ELEMENT_TYPE_ATTRIBUTE = "elementType"
# Largest deviation of the section axes (unit vectors) from horizontal u and vertical v.
AXIS_TOLERANCE = 1e-9
_UNIT_ABBREVIATIONS = {"metre": "m", "meter": "m", "foot": "ft", "US survey foot": "ftUS"}
_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SectionsSource:
    """An Evo ``geological-sections`` object and a way to load its data tables."""

    geoscience_object: GeologicalSectionsObject
    load_table: TableLoader
    object_id: Optional[str] = None
    version_id: Optional[str] = None


@dataclass
class AgsiExport:
    """An AGSi package in memory: main JSON document and supporting geometry files."""

    title: str
    main_file: str
    document: dict[str, Any]
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def index(self) -> dict[str, Any]:
        return {"agsiIndex": {"filePath": self.main_file, "schemaVersion": AGSI_VERSION, "title": self.title}}

    def write(self, filepath: Union[str, Path]) -> None:
        """Write the package as a ``.agsi`` ZIP archive."""
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(INDEX_FILE_NAME, _json_bytes(self.index))
            archive.writestr(self.main_file, _json_bytes(self.document))
            for name, data in self.files.items():
                archive.writestr(name, data)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# --- polygon -> profiles -------------------------------------------------------------------


def assemble_ring(vertices: FloatArray, chunks: IntArray, parts: Sequence[Any], name: str) -> FloatArray:
    """Closed boundary (first vertex repeated at the end) of an Evo volume.

    :param parts: ``ReversibleIndex`` parts (``index`` into ``chunks``, ``reversed``) in ring order.
    :raise AGSIExporterException: If a part refers to a missing chunk, or the parts are not
        connected end to start exactly or do not close.
    """
    ring = assemble_chain(vertices, chunks, parts, name)
    if not np.array_equal(ring[-1], ring[0]):
        raise AGSIExporterException(f"{name}: the volume parts do not form a closed ring.")
    return ring


def assemble_chain(vertices: FloatArray, chunks: IntArray, parts: Sequence[Any], name: str) -> FloatArray:
    """Vertices of connected line chunks (each part starts exactly where the previous one ends)."""
    if not parts:
        raise AGSIExporterException(f"{name}: no line parts.")
    pieces: list[FloatArray] = []
    for part in parts:
        index = int(part.index)
        if not 0 <= index < len(chunks):
            raise AGSIExporterException(f"{name}: part refers to line chunk {index}, but there are {len(chunks)}.")
        offset, count = (int(value) for value in chunks[index])
        if count < 1 or offset < 0 or offset + count > len(vertices):
            raise AGSIExporterException(f"{name}: line chunk {index} is outside the line vertices.")
        piece = vertices[offset : offset + count]
        if part.reversed:
            piece = piece[::-1]
        if pieces and not np.array_equal(pieces[-1][-1], piece[0]):
            raise AGSIExporterException(
                f"{name}: line chunk {index} does not start where the previous part ends; the parts must be connected."
            )
        pieces.append(piece)
    return np.concatenate([pieces[0], *(piece[1:] for piece in pieces[1:])])


def ring_to_profiles(ring: npt.ArrayLike, name: str) -> tuple[FloatArray, FloatArray]:
    """Split a closed section polygon into its top and bottom lines, both in increasing chainage u.

    The split is accepted only when it is unique: after removing repeated consecutive vertices,

    * the vertices at the minimum u, and those at the maximum u, are each one contiguous run (a
      single vertex, or a vertical edge whose v changes monotonically);
    * between those runs the ring consists of two chains with strictly increasing (and
      decreasing) u, i.e. no vertical steps, overhangs or folds inside the polygon;
    * the polygon has a non-zero signed area, which identifies the upper chain (the ring may run
      either way round);
    * the top is nowhere below the bottom, tested exactly as when the profiles are read again.

    Vertices inside a vertical end edge are dropped: they are collinear with its end points.

    :raise UnsupportedSectionGeometryError: If any of the conditions above is not met.
    :raise AGSIExporterException: For non-finite coordinates.
    """
    points = np.asarray(ring, dtype=np.float64).reshape(-1, 2) + 0.0
    if not np.all(np.isfinite(points)):
        raise AGSIExporterException(f"{name}: non-finite coordinates.")
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points):
        points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]
    if len(points) < 3:
        raise UnsupportedSectionGeometryError(f"{name}: the polygon has fewer than three distinct vertices.")
    u = points[:, 0]
    low, high = float(u.min()), float(u.max())
    if low == high:
        raise UnsupportedSectionGeometryError(f"{name}: the polygon has zero width in chainage.")

    left_start, left_end = _single_run(u == low, name, "minimum")
    right_start, right_end = _single_run(u == high, name, "maximum")
    for start, end, side in ((left_start, left_end, "left"), (right_start, right_end, "right")):
        edge = _cyclic_slice(points, start, end)[:, 1]
        steps = np.diff(edge)
        if len(edge) > 2 and not (np.all(steps > 0) or np.all(steps < 0)):
            raise UnsupportedSectionGeometryError(f"{name}: the {side} end of the polygon folds back on itself.")

    forward = _cyclic_slice(points, left_end, right_start)
    backward = _cyclic_slice(points, right_end, left_start)[::-1]
    for chain in (forward, backward):
        if np.any(np.diff(chain[:, 0]) <= 0):
            raise UnsupportedSectionGeometryError(
                f"{name}: the polygon boundary is not one top and one bottom line with strictly increasing "
                "chainage (it has a vertical step, an overhang or a fold); it cannot be written as AGSi "
                "top and bottom profiles."
            )
    area = _twice_signed_area(points)
    if area == 0:
        raise UnsupportedSectionGeometryError(f"{name}: the polygon has zero area.")
    # Anticlockwise (positive area): the interior is to the left of the forward chain, i.e. above it.
    top, bottom = (backward, forward) if area > 0 else (forward, backward)
    try:
        build_section_geometry([SectionUnit(key=name, top=top, bottom=bottom)])
    except AGSIInvalidDataError as e:
        raise UnsupportedSectionGeometryError(f"{name}: {e}") from e
    return top, bottom


def _single_run(mask: npt.NDArray[np.bool_], name: str, which: str) -> tuple[int, int]:
    """First and last index of the single cyclic run of ``True`` in ``mask``."""
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    ends = np.flatnonzero(mask & ~np.roll(mask, -1))
    if len(starts) != 1 or len(ends) != 1:
        raise UnsupportedSectionGeometryError(
            f"{name}: the polygon reaches its {which} chainage at {len(starts)} separate places; it cannot "
            "be written as AGSi top and bottom profiles."
        )
    return int(starts[0]), int(ends[0])


def _cyclic_slice(points: FloatArray, start: int, end: int) -> FloatArray:
    if end >= start:
        return points[start : end + 1]
    return np.concatenate([points[start:], points[: end + 1]])


def _twice_signed_area(points: FloatArray) -> Fraction:
    """Exact shoelace sum of the polygon (positive when anticlockwise)."""
    exact = [(Fraction(float(x)), Fraction(float(y))) for x, y in points]
    return sum(
        (x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(exact, exact[1:] + exact[:1])),
        Fraction(0),
    )


# --- section placement ---------------------------------------------------------------------


def section_axes(section: Any, name: str) -> tuple[FloatArray, FloatArray]:
    """World origin and horizontal unit direction of increasing u of an Evo section.

    :raise UnsupportedSectionGeometryError: Unless section u is horizontal and v points straight up
        (within :data:`AXIS_TOLERANCE`), i.e. the section is a vertical plane.
    """
    origin = np.asarray(section.origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise AGSIExporterException(f"Section '{name}': origin must be three finite coordinates.")
    rotation = section.rotation
    angles = [-rotation.dip_azimuth, -rotation.dip, -rotation.pitch]
    # Inverse of evo.data_converters.common.utils.convert_rotation: columns are the u, v, w axes.
    matrix = Rotation.from_euler("ZXZ", angles, degrees=True).as_matrix()
    u_axis, v_axis = matrix[:, 0], matrix[:, 1]
    if abs(u_axis[2]) > AXIS_TOLERANCE or np.max(np.abs(v_axis - (0.0, 0.0, 1.0))) > AXIS_TOLERANCE:
        raise UnsupportedSectionGeometryError(
            f"Section '{name}' (dip_azimuth {rotation.dip_azimuth}, dip {rotation.dip}, pitch {rotation.pitch}) "
            "is not a vertical plane with u horizontal and v pointing up; only such sections can be written "
            "as AGSi chainage/elevation profiles."
        )
    direction = u_axis[:2] / math.hypot(u_axis[0], u_axis[1])
    return origin, direction


# --- formatting ----------------------------------------------------------------------------


def _number(value: float) -> str:
    """Shortest decimal that reads back as the same float, without an exponent."""
    return str(np.format_float_positional(float(value) + 0.0, unique=True, trim="-"))


def linestring_wkt(points: npt.ArrayLike) -> str:
    coordinates = ", ".join(" ".join(_number(value) for value in point) for point in np.asarray(points))
    return f"LINESTRING ({coordinates})"


def _colour_rgb(colour: Optional[int]) -> Optional[str]:
    """``#RRGGBB`` from an Evo material colour (alpha, blue, green, red from the highest byte)."""
    if colour is None:
        return None
    red, green, blue = colour & 0xFF, (colour >> 8) & 0xFF, (colour >> 16) & 0xFF
    return f"#{red:02X}{green:02X}{blue:02X}"


def _slug(text: str) -> str:
    return _UNSAFE_FILE_CHARS.sub("_", text).strip("_.") or "item"


def _unique(base: str, used: set[str]) -> str:
    candidate, number = base, 2
    while candidate in used:
        candidate, number = f"{base} ({number})", number + 1
    used.add(candidate)
    return candidate


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _number_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# --- document assembly ---------------------------------------------------------------------


@dataclass
class _Piece:
    """One Evo volume as top and bottom profiles."""

    index: int
    volume: Any
    top: FloatArray
    bottom: FloatArray


@dataclass
class _Unit:
    """One AGSi element: one or more Evo volumes, and the element's complete top and bottom lines."""

    element_id: str
    pieces: list[_Piece]
    top: FloatArray
    bottom: FloatArray
    row: "_Row"
    part: str = ""
    omit_top: bool = False
    omit_bottom: bool = False

    @property
    def volume(self) -> Any:
        return self.pieces[0].volume


@dataclass
class _Row:
    """Element fields and parameters of one Evo volume, from the volume attributes."""

    element_id: Optional[str] = None
    element_type: Optional[str] = None
    parameters: list[dict[str, Any]] = field(default_factory=list)


class _Builder:
    def __init__(self, title: str) -> None:
        self.title = title
        self.systems: list[dict[str, Any]] = []
        self.models: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.remarks: list[str] = []
        self._system_by_crs: dict[str, tuple[str, str]] = {}
        self._model_ids: set[str] = set()
        self._system_ids: set[str] = set()
        self._alignment_ids: set[str] = set()
        self._file_names: set[str] = set()
        self._profile_uris: dict[bytes, str] = {}

    # -- objects --

    def add_object(self, source: SectionsSource) -> None:
        go = source.geoscience_object
        name = go.name
        extension = (go.extensions or {}).get("agsi")
        extension = extension if isinstance(extension, dict) else {}
        reference = ", ".join(
            f"{label} {value}" for label, value in (("ID", source.object_id), ("version", source.version_id)) if value
        )
        self.remarks.append(f"'{name}'" + (f" ({reference})" if reference else ""))

        vertices, chunks = _line_geometry(go, source.load_table, name)
        rows = _volume_rows(go, source.load_table, extension)
        provenance = _contact_provenance(go, extension)
        materials = {material.key: material for material in go.materials or []}
        folders = _section_folders(go.folders or [])
        system_id, units = self._xy_system(go.coordinate_reference_system)
        single = len(go.sections) == 1

        alignments: list[dict[str, Any]] = []
        for index, section in enumerate(go.sections):
            section_name = section.name or f"Section {index + 1}"
            label = f"Object '{name}', section '{section_name}'"
            volume_indices = _range_indices(section.volumes, len(go.volumes), f"{label}: volumes")
            if not volume_indices:
                logger.warning(f"{label} has no volumes; it is not exported.")
                continue
            origin, direction = section_axes(section, section_name)
            units_in_order = _ordered_units(go, section, volume_indices, vertices, chunks, rows, provenance, label)
            boundaries = _boundary_lines(go, section, vertices, chunks)
            boundary = _omit_generated_lines(units_in_order, boundaries, origin[2])
            alignment = self._add_section_model(
                go,
                section,
                section_name,
                single,
                extension,
                folders.get(index, []),
                units_in_order,
                materials,
                origin,
                direction,
                boundary,
                units,
            )
            alignments.append(alignment)

        if not alignments:
            raise AGSIExporterException(f"Object '{name}' has no section volumes to export.")
        unused = len(go.surfaces)
        if unused:
            logger.info(
                f"Object '{name}': {unused} surfaces are not exported; contacts are written as profile files "
                "shared by the elements above and below them."
            )
        self.models.append(
            {
                "modelID": _unique(f"{name} alignment plan", self._model_ids),
                "modelName": f"{name} alignments (plan)",
                "description": "2D map (XY) model holding the alignments of the section models exported from "
                f"Evo object '{name}'; it contains no model elements.",
                "coordSystemID": system_id,
                "agsiModelAlignment": alignments,
            }
        )

    def _add_section_model(
        self,
        go: Any,
        section: Any,
        section_name: str,
        single: bool,
        extension: dict[str, Any],
        folder: list[str],
        units: list[_Unit],
        materials: dict[str, Any],
        origin: FloatArray,
        direction: FloatArray,
        boundary: dict[str, float],
        axis_units: str,
    ) -> dict[str, Any]:
        own = extension if single else {}
        model_id = _unique(
            _text(own.get("modelID")) or (go.name if single else f"{go.name} - {section_name}"), self._model_ids
        )
        alignment_id = _unique(_text(own.get("alignmentID")) or f"{model_id} alignment", self._alignment_ids)
        alignment_name = _text(own.get("alignmentName")) or (folder[-1] if folder else section_name)
        system_id = _unique(_text(own.get("coordSystemID")) or f"{model_id} XZ", self._system_ids)
        self.systems.append(
            {
                "systemID": system_id,
                "description": f"2D vertical section along alignment '{alignment_id}': X = chainage along the "
                "alignment, Z = elevation",
                "systemType": "XZ",
                "axisNameX": "Chainage",
                "axisNameZ": "Elevation",
                "axisUnitsXY": axis_units,
                "axisUnitsZ": axis_units,
            }
        )

        start, length = _chainage_range(units, own, model_id)
        ends = np.array([start, start + length])
        alignment_xy = origin[:2] + np.outer(ends, direction)
        alignment_uri = self._add_file(f"Alignments/{_slug(alignment_id)}.wkt", linestring_wkt(alignment_xy))

        model_slug = _slug(model_id)
        element_ids: set[str] = set()
        elements = [
            self._element(unit, _unique(unit.element_id + unit.part, element_ids), materials, origin[2], model_slug)
            for unit in units
        ]

        model: dict[str, Any] = {
            "modelID": model_id,
            "modelName": _text(own.get("modelName")) or section_name,
            "description": section.description or go.description,
            "coordSystemID": system_id,
            "alignmentID": alignment_id,
            "agsiModelElement": elements,
        }
        if boundary:
            model["agsiModelBoundary"] = {
                "description": "Elevation of the horizontal boundary lines of the Evo section that the units "
                "without a top or bottom profile extend to.",
                **boundary,
            }
        remarks = [f"Exported from section '{section_name}' of Evo geological-sections object '{go.name}'."]
        if folder:
            remarks.append(f"Evo folder: {' / '.join(folder)}.")
        model["remarks"] = " ".join(remarks)
        self.models.append({key: value for key, value in model.items() if value is not None})

        alignment: dict[str, Any] = {
            "alignmentID": alignment_id,
            "alignmentName": alignment_name,
            "description": f"Straight alignment of section '{section_name}'",
            "agsiGeometry": {
                "description": "Section line (XY)",
                "geometryType": "Line",
                "fileFormat": "WKT",
                "fileURI": alignment_uri,
            },
            "startChainage": start,
        }
        return alignment

    def _element(
        self,
        unit: _Unit,
        element_id: str,
        materials: dict[str, Any],
        origin_z: float,
        model_slug: str,
    ) -> dict[str, Any]:
        volume, row = unit.volume, unit.row
        geometry: dict[str, Any] = {"description": volume.name}
        for which, line, omit in (("Top", unit.top, unit.omit_top), ("Bottom", unit.bottom, unit.omit_bottom)):
            if omit:
                continue
            label = "Top" if which == "Top" else "Base"
            uri = self._add_profile(line, origin_z, f"Profiles/{model_slug}", f"{_slug(element_id)}-{label}")
            geometry[f"agsiGeometry{which}"] = {
                "description": f"{label} of {volume.name}",
                "geometryType": "Profile",
                "fileFormat": "WKT",
                "fileURI": uri,
            }
        material = materials.get(volume.material_key) if volume.material_key is not None else None
        element: dict[str, Any] = {
            "elementID": element_id,
            "elementName": volume.name if volume.name and volume.name != element_id else None,
            "description": volume.description,
            "elementType": row.element_type,
            "geometryObject": "agsiGeometryAreaFromLines",
            "agsiGeometry": geometry,
            "agsiDataParameterValue": row.parameters or None,
            "colourRGB": _colour_rgb(material.color) if material is not None else None,
            "remarks": f"One of several separate polygons of '{unit.element_id}' in the Evo section."
            if unit.part
            else None,
        }
        return {key: value for key, value in element.items() if value is not None}

    # -- files --

    def _add_file(self, name: str, text: str) -> str:
        stem, dot, suffix = name.rpartition(".")
        candidate, number = name, 2
        while candidate.lower() in self._file_names:
            candidate, number = f"{stem}-{number}{dot}{suffix}", number + 1
        self._file_names.add(candidate.lower())
        self.files[candidate] = text.encode("utf-8")
        return candidate

    def _add_profile(self, line: FloatArray, origin_z: float, folder: str, stem: str) -> str:
        """Write a (chainage, elevation) profile once; identical profiles share one file."""
        profile = np.column_stack([line[:, 0], line[:, 1] + origin_z]) + 0.0
        key = np.ascontiguousarray(profile).tobytes()
        uri = self._profile_uris.get(key)
        if uri is None:
            uri = self._profile_uris[key] = self._add_file(f"{folder}/{stem}.wkt", linestring_wkt(profile))
        return uri

    # -- coordinate systems --

    def _xy_system(self, crs: Any) -> tuple[str, str]:
        """ID and axis units of the XY coordinate system for an Evo CRS (one system per distinct CRS)."""
        crs_key = repr(crs)
        if crs_key in self._system_by_crs:
            return self._system_by_crs[crs_key]
        system: dict[str, Any] = {"systemType": "XY"}
        parsed: Optional[CRS] = None
        if isinstance(crs, Crs_V1_0_1_EpsgCode):
            system["globalXYSystem"] = f"EPSG:{crs.epsg_code}"
            system["remarks"] = "From the coordinate_reference_system of the Evo object."
            parsed = _parse_crs(lambda: CRS.from_epsg(crs.epsg_code))
        elif isinstance(crs, Crs_V1_0_1_OgcWkt):
            system["remarks"] = f"Evo coordinate_reference_system (OGC WKT): {crs.ogc_wkt}"
            parsed = _parse_crs(lambda: CRS.from_wkt(crs.ogc_wkt))
        else:
            system["remarks"] = "The Evo object has no coordinate reference system; axis units are unknown."
            logger.warning("An exported Evo object has no coordinate reference system; axis units are 'unknown'.")
        units = "unknown"
        if parsed is not None:
            system["systemNameXY"] = parsed.name
            axes = parsed.axis_info[:2]
            if len(axes) == 2:
                system["axisNameX"], system["axisNameY"] = axes[0].name, axes[1].name
            unit_names = {axis.unit_name for axis in axes}
            if len(unit_names) == 1 and not parsed.is_geographic:
                unit = unit_names.pop()
                units = _UNIT_ABBREVIATIONS.get(unit, unit)
        system["axisUnitsXY"] = units
        system["axisUnitsZ"] = units
        system_id = _unique(
            "Evo-XY" if not self._system_by_crs else f"Evo-XY-{len(self._system_by_crs) + 1}", self._system_ids
        )
        self.systems.append({"systemID": system_id, **system})
        self._system_by_crs[crs_key] = (system_id, units)
        return system_id, units

    # -- output --

    def build(self, main_file: str) -> AgsiExport:
        document = {
            "agsSchema": {"name": "AGSi", "version": AGSI_VERSION, "link": AGSI_LINK},
            "agsFile": {
                "title": self.title,
                "description": f"AGSi {AGSI_VERSION} export of Evo geological-sections objects: "
                + "; ".join(self.remarks)
                + ".",
                "producedBy": PRODUCED_BY,
                "remarks": "Section models hold one agsiGeometryAreaFromLines element per AGSi element recorded "
                "in the Evo object (else per group of touching Evo volumes with one element ID); their "
                "alignments are straight lines in a separate plan model. Profiles shared by neighbouring "
                "elements are written once and referenced by both.",
            },
            "agsProject": {"projectName": self.title, "agsProjectCoordinateSystem": self.systems},
            "agsiModel": self.models,
        }
        return AgsiExport(title=self.title, main_file=main_file, document=document, files=self.files)


def _parse_crs(factory: Callable[[], CRS]) -> Optional[CRS]:
    try:
        return factory()
    except CRSError as e:
        logger.warning(f"Coordinate reference system not recognised ({e}); axis units are 'unknown'.")
        return None


def build_agsi_export(sources: Sequence[SectionsSource], title: str, main_file: Optional[str] = None) -> AgsiExport:
    """Convert Evo ``geological-sections`` objects to an AGSi 1.0.1 package (in memory).

    :param sources: Objects to export, with a loader for their data tables.
    :param title: AGSi ``agsFile.title`` and ``agsProject.projectName``.
    :param main_file: Name of the main JSON file in the package; defaults to ``<title>.json``.
    :raise UnsupportedSectionGeometryError: If a section is not vertical, or a volume cannot be
        written as top and bottom profiles (see :func:`ring_to_profiles`).
    :raise AGSIExporterException: If there is nothing to export or the object data is invalid.
    """
    if not sources:
        raise AGSIExporterException("No objects to export.")
    builder = _Builder(title)
    for source in sources:
        builder.add_object(source)
    return builder.build(main_file or f"{_slug(title)}.json")


# --- object data ---------------------------------------------------------------------------


def _line_geometry(go: Any, load_table: TableLoader, name: str) -> tuple[FloatArray, IntArray]:
    geometry = go.line_geometry
    table = load_table(geometry.vertices)
    if table.num_columns < 2:
        raise AGSIExporterException(f"Object '{name}': line geometry vertices need two columns (u, v).")
    vertices = np.column_stack([table.column(k).to_numpy(zero_copy_only=False) for k in (0, 1)]).astype(np.float64)
    chunk_table = load_table(geometry.chunks)
    if chunk_table.num_columns < 2:
        raise AGSIExporterException(f"Object '{name}': line geometry chunks need two columns (offset, count).")
    chunks = np.column_stack([chunk_table.column(k).to_numpy(zero_copy_only=False) for k in (0, 1)]).astype(np.int64)
    if geometry.indices is not None:
        indices = load_table(geometry.indices).column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise AGSIExporterException(f"Object '{name}': line geometry indices are outside the vertices.")
        vertices = vertices[indices]
    return vertices, chunks.reshape(-1, 2)


def _range_indices(indices_range: Any, total: int, what: str) -> list[int]:
    if indices_range is None:
        return []
    offset, count = int(indices_range.offset), int(indices_range.count)
    if offset < 0 or count < 0 or offset + count > total:
        raise AGSIExporterException(f"{what}: range {offset}..{offset + count} is outside the {total} items.")
    return list(range(offset, offset + count))


def _ordered_units(
    go: Any,
    section: Any,
    volume_indices: list[int],
    vertices: FloatArray,
    chunks: IntArray,
    rows: list[_Row],
    provenance: Optional[list[dict[str, Any]]],
    label: str,
) -> list[_Unit]:
    """AGSi elements of a section, ordered by ``layer_order`` where given.

    With contact provenance (``extensions.agsi.surfaces``) and a section that lists its surfaces,
    the volumes of each element are reconciled with its top and bottom surfaces; otherwise volumes
    are grouped by element ID.
    """
    pieces = []
    for index in volume_indices:
        volume = go.volumes[index]
        volume_label = f"{label}, volume '{volume.name}'"
        if getattr(volume, "internal_rings", None):
            raise UnsupportedSectionGeometryError(
                f"{volume_label} has internal rings (holes); AGSi areas from lines cannot have holes."
            )
        ring = assemble_ring(vertices, chunks, volume.parts, volume_label)
        top, bottom = ring_to_profiles(ring, volume_label)
        pieces.append(_Piece(index, volume, top, bottom))
    if provenance is not None and section.surfaces is not None:
        units = _units_from_provenance(go, section, pieces, vertices, chunks, rows, provenance, label)
    else:
        units = _units_from_volumes(pieces, rows, label)
    order = {layer: position for position, layer in enumerate(getattr(go, "layer_order", None) or [])}
    if order:
        units.sort(key=lambda unit: order.get(_layer(unit.volume), len(order)))
    return units


def _layer(volume: Any) -> Optional[str]:
    return _text(volume.layer) or _text(volume.material_key)


def _contact_provenance(go: Any, extension: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """``extensions.agsi.surfaces`` written by the importer, checked against the object's surfaces.

    :raise AGSIExporterException: If it is present but does not describe the current surfaces.
    """
    if "surfaces" not in extension:
        return None
    entries = extension["surfaces"]
    hint = (
        "it no longer matches the object, which was probably edited since it was imported from AGSi. Remove "
        "extensions.agsi.surfaces to export every volume as a separate element"
    )
    if not isinstance(entries, list) or len(entries) != len(go.surfaces):
        raise AGSIExporterException(
            f"Object '{go.name}': extensions.agsi.surfaces describes "
            f"{len(entries) if isinstance(entries, list) else 'no'} surfaces but the object has "
            f"{len(go.surfaces)}; {hint}."
        )
    for index, (entry, surface) in enumerate(zip(entries, go.surfaces)):
        valid = (
            isinstance(entry, dict)
            and entry.get("name") == surface.name
            and all(
                isinstance(entry.get(key), list) and all(isinstance(item, str) for item in entry[key])
                for key in ("topOf", "bottomOf")
            )
        )
        if not valid:
            raise AGSIExporterException(
                f"Object '{go.name}': extensions.agsi.surfaces[{index}] does not describe surface "
                f"'{surface.name}'; {hint}."
            )
    return entries


def _units_from_provenance(
    go: Any,
    section: Any,
    pieces: list[_Piece],
    vertices: FloatArray,
    chunks: IntArray,
    rows: list[_Row],
    provenance: list[dict[str, Any]],
    label: str,
) -> list[_Unit]:
    """One element per layer: its top and bottom surfaces, checked against its volumes.

    The volumes of an element must be exactly the polygons that its current top and bottom lines
    enclose (as the importer builds them), so geometry edits made consistently in Evo are
    exported, while deleted, added or reshaped volumes that disagree with the surfaces are
    rejected rather than replaced by the surface lines.
    """
    tops: dict[str, list[int]] = {}
    bottoms: dict[str, list[int]] = {}
    for index in _range_indices(section.surfaces, len(go.surfaces), f"{label}: surfaces"):
        for key in provenance[index]["topOf"]:
            tops.setdefault(key, []).append(index)
        for key in provenance[index]["bottomOf"]:
            bottoms.setdefault(key, []).append(index)
    groups: dict[str, list[_Piece]] = {}
    for piece in pieces:
        key = _layer(piece.volume)
        if key is None or (key not in tops and key not in bottoms):
            raise AGSIExporterException(
                f"{label}, volume '{piece.volume.name}': layer '{key}' is not the top or bottom of any surface "
                "in extensions.agsi.surfaces, so its AGSi element cannot be reconstructed. Remove "
                "extensions.agsi.surfaces to export every volume as a separate element."
            )
        groups.setdefault(key, []).append(piece)

    def line(key: str, which: str, found: dict[str, list[int]]) -> tuple[FloatArray, str]:
        if len(found.get(key, [])) != 1:
            raise AGSIExporterException(
                f"{label}: extensions.agsi.surfaces names {len(found.get(key, []))} {which} surfaces for element "
                f"'{key}'; exactly one is needed."
            )
        surface = go.surfaces[found[key][0]]
        chain = assemble_chain(vertices, chunks, surface.parts, f"{label}, surface '{surface.name}'")
        try:
            return profile_to_section(chain, f"{label}, surface '{surface.name}'"), surface.name
        except AGSIInvalidDataError as e:
            raise UnsupportedSectionGeometryError(str(e)) from e

    units = []
    for key in dict.fromkeys([*groups, *tops, *bottoms]):
        (top, top_name), (bottom, bottom_name) = line(key, "top", tops), line(key, "bottom", bottoms)
        group = groups.get(key, [])
        expected = _enclosed_pieces(top, bottom, f"{label}, element '{key}'")
        if expected != sorted(_piece_key(piece.top, piece.bottom) for piece in group):
            raise UnsupportedSectionGeometryError(
                f"{label}: the {len(group)} volumes of element '{key}' are not the {len(expected)} polygons "
                f"between its top surface '{top_name}' and bottom surface '{bottom_name}' recorded in "
                "extensions.agsi.surfaces; the object was edited inconsistently with that provenance. Remove "
                "extensions.agsi.surfaces to export every volume as a separate element."
            )
        if not group:
            logger.warning(f"{label}: element '{key}' has zero thickness everywhere (no volume); it is not exported.")
            continue
        row = _group_row(group, rows)
        if row is None:
            raise AGSIExporterException(
                f"{label}: the volumes of element '{key}' have different names, descriptions, materials or "
                "attribute values; they cannot be written as one AGSi element."
            )
        units.append(_Unit(row.element_id or key, group, top, bottom, row))
    return units


def _units_from_volumes(pieces: list[_Piece], rows: list[_Row], label: str) -> list[_Unit]:
    """Elements from volumes grouped by element ID, without contact provenance.

    Volumes of one element are merged only when they meet end to end at single-vertex pinch-outs
    and the merged lines enclose exactly those volumes; nothing is added between them. Otherwise
    each volume becomes its own ``"<element ID> (part n)"`` element.
    """
    groups: dict[str, list[_Piece]] = {}
    for piece in pieces:
        volume, row = piece.volume, rows[piece.index]
        base = row.element_id or _layer(volume) or volume.name
        groups.setdefault(base, []).append(piece)
    units = []
    for base, group in groups.items():
        if len(group) == 1:
            units.append(_Unit(base, group, group[0].top, group[0].bottom, rows[group[0].index]))
            continue
        group_row = _group_row(group, rows)
        merged = _merge_touching(group, f"{label}, element '{base}'") if group_row is not None else None
        if group_row is not None and merged is not None:
            units.append(_Unit(base, group, merged[0], merged[1], group_row))
            continue
        logger.warning(
            f"{label}: element '{base}' has {len(group)} separate volumes that do not meet at single-vertex "
            "pinch-outs, and the object has no contact provenance (extensions.agsi.surfaces); each volume is "
            "written as a separate '(part n)' element."
        )
        units.extend(
            _Unit(base, [piece], piece.top, piece.bottom, rows[piece.index], part=f" (part {number})")
            for number, piece in enumerate(group, start=1)
        )
    return units


def _merge_touching(group: list[_Piece], label: str) -> Optional[tuple[FloatArray, FloatArray]]:
    """Top and bottom lines through volumes that meet exactly at pinch-out vertices, if they do."""
    ordered = sorted(group, key=lambda piece: float(piece.top[0, 0]))
    for before, after in zip(ordered, ordered[1:]):
        joint = before.top[-1]
        if not (
            np.array_equal(before.bottom[-1], joint)
            and np.array_equal(after.top[0], joint)
            and np.array_equal(after.bottom[0], joint)
        ):
            return None
    top = np.concatenate([ordered[0].top, *(piece.top[1:] for piece in ordered[1:])])
    bottom = np.concatenate([ordered[0].bottom, *(piece.bottom[1:] for piece in ordered[1:])])
    try:
        expected = _enclosed_pieces(top, bottom, label)
    except UnsupportedSectionGeometryError:
        return None
    if expected != sorted(_piece_key(piece.top, piece.bottom) for piece in group):
        return None
    return top, bottom


def _enclosed_pieces(top: FloatArray, bottom: FloatArray, label: str) -> list[tuple[bytes, bytes]]:
    """Profiles of the polygons the importer builds between ``top`` and ``bottom`` (sorted keys)."""
    try:
        geometry = build_section_geometry([SectionUnit(key=label, top=top, bottom=bottom)])
    except AGSIInvalidDataError as e:
        raise UnsupportedSectionGeometryError(f"{label}: {e}") from e
    return sorted(_piece_key(*ring_to_profiles(geometry.ring(volume), label)) for volume in geometry.volumes)


def _piece_key(top: FloatArray, bottom: FloatArray) -> tuple[bytes, bytes]:
    return (
        np.ascontiguousarray(top, dtype=np.float64).tobytes(),
        np.ascontiguousarray(bottom, dtype=np.float64).tobytes(),
    )


def _group_row(group: list[_Piece], rows: list[_Row]) -> Optional[_Row]:
    """Shared element fields of the volumes of one element, or ``None`` if they differ."""
    first = group[0]
    for piece in group[1:]:
        volume = piece.volume
        if (
            rows[piece.index] != rows[first.index]
            or volume.name != first.volume.name
            or volume.description != first.volume.description
            or volume.material_key != first.volume.material_key
        ):
            return None
    return rows[first.index]


def _boundary_lines(go: Any, section: Any, vertices: FloatArray, chunks: IntArray) -> set[bytes]:
    """Vertices (as bytes) of the horizontal ``BoundarySurface`` lines of a section."""
    lines = set()
    for index in _range_indices(section.surfaces, len(go.surfaces), f"Section '{section.name}': surfaces"):
        surface = go.surfaces[index]
        if surface.feature != BOUNDARY_FEATURE:
            continue
        try:
            line = assemble_chain(vertices, chunks, surface.parts, "surface") + 0.0
        except AGSIExporterException:
            continue
        if line[0, 0] > line[-1, 0]:
            line = line[::-1]
        if np.all(line[:, 1] == line[0, 1]):
            lines.add(np.ascontiguousarray(line).tobytes())
    return lines


def _omit_generated_lines(units: list[_Unit], boundaries: set[bytes], origin_z: float) -> dict[str, float]:
    """Mark horizontal boundary-surface tops/bottoms to be written as model boundary elevations.

    Mirrors the importer, which generates such lines for AGSi areas without a top or bottom
    profile. Only done when all such bottoms (tops) of the section are at one elevation.
    """
    boundary: dict[str, float] = {}
    for attribute, key in (("bottom", "bottomElevation"), ("top", "topElevation")):
        matches = [unit for unit in units if np.ascontiguousarray(getattr(unit, attribute)).tobytes() in boundaries]
        elevations = {float(getattr(unit, attribute)[0, 1]) for unit in matches}
        if len(elevations) == 1:
            for unit in matches:
                setattr(unit, f"omit_{attribute}", True)
            boundary[key] = elevations.pop() + origin_z
        elif elevations:
            logger.warning(
                f"Horizontal boundary {attribute} lines at several elevations; they are written as {attribute} "
                "profiles."
            )
    for unit in units:
        if unit.omit_top and unit.omit_bottom:
            unit.omit_top = False
    return boundary


def _chainage_range(units: list[_Unit], extension: dict[str, Any], name: str) -> tuple[float, float]:
    """Start chainage and length of the alignment: from the AGSi extension when it covers the profiles."""
    low = min(float(line[0, 0]) for unit in units for line in (unit.top, unit.bottom))
    high = max(float(line[-1, 0]) for unit in units for line in (unit.top, unit.bottom))
    start, length = _number_or_none(extension.get("startChainage")), _number_or_none(extension.get("alignmentLength"))
    if start is not None and length is not None and length > 0:
        if start <= low and high <= start + length:
            return start, length
        logger.warning(
            f"Model '{name}': the profiles extend beyond the alignment recorded in the object extensions; the "
            "alignment is made to span the profiles."
        )
    return low, high - low


def _section_folders(folders: list[Any], path: Optional[list[str]] = None) -> dict[int, list[str]]:
    """Folder path of each section index."""
    path = path or []
    found: dict[int, list[str]] = {}
    for folder in folders:
        folder_path = [*path, folder.name]
        for item in folder.items:
            if hasattr(item, "section_index"):
                found.setdefault(int(item.section_index), folder_path)
            else:
                for index, nested in _section_folders([item], path=folder_path).items():
                    found.setdefault(index, nested)
    return found


def _volume_rows(go: Any, load_table: TableLoader, extension: dict[str, Any]) -> list[_Row]:
    """Element ID, type and AGSi parameters of every volume, from the volume attributes.

    Continuous and integer attributes become ``valueNumeric``, string and category attributes
    ``valueText``; missing values are skipped. Attribute names map back to codeID and caseID via
    the ``parameterColumns`` the importer records, else the name is the codeID. Other attribute
    types are not exported.
    """
    rows = [_Row() for _ in go.volumes]
    columns = extension.get("parameterColumns")
    columns = columns if isinstance(columns, dict) else {}
    for attribute in go.volume_attributes or []:
        name = attribute.name
        values = _attribute_values(attribute, load_table)
        if values is None:
            logger.warning(
                f"Object '{go.name}': volume attribute '{name}' ({type(attribute).__name__}) is not exported."
            )
            continue
        if len(values) != len(rows):
            raise AGSIExporterException(
                f"Object '{go.name}': volume attribute '{name}' has {len(values)} values for {len(rows)} volumes."
            )
        if name in (ELEMENT_ID_ATTRIBUTE, ELEMENT_TYPE_ATTRIBUTE) and isinstance(attribute, StringAttribute_V1_1_0):
            for row, value in zip(rows, values):
                setattr(row, "element_id" if name == ELEMENT_ID_ATTRIBUTE else "element_type", _text(value))
            continue
        source = columns.get(name)
        code = source.get("codeID") if isinstance(source, dict) else None
        case = source.get("caseID") if isinstance(source, dict) else None
        for row, value in zip(rows, values):
            if value is None:
                continue
            parameter: dict[str, Any] = {"codeID": _text(code) or name}
            if _text(case):
                parameter["caseID"] = case
            parameter["valueNumeric" if isinstance(value, float) else "valueText"] = value
            row.parameters.append(parameter)
    return rows


def _attribute_values(attribute: Any, load_table: TableLoader) -> Optional[list[Any]]:
    """Per-volume values (float, str, or None when missing) of a supported attribute, else None."""
    if isinstance(attribute, ContinuousAttribute_V1_1_0):
        missing = {float(value) for value in attribute.nan_description.values or []}
        numbers = _column(load_table(attribute.values))
        return [
            None if value is None or not math.isfinite(value) or value in missing else float(value) for value in numbers
        ]
    if isinstance(attribute, IntegerAttribute_V1_1_0):
        missing = {int(value) for value in attribute.nan_description.values or []}
        return [
            None if value is None or value in missing else float(value)
            for value in _column(load_table(attribute.values))
        ]
    if isinstance(attribute, StringAttribute_V1_1_0):
        return [value if isinstance(value, str) else None for value in _column(load_table(attribute.values))]
    if isinstance(attribute, CategoryAttribute_V1_1_0):
        lookup_table = load_table(attribute.table)
        lookup = dict(zip(_column(lookup_table, 0), _column(lookup_table, 1)))
        missing = {int(value) for value in attribute.nan_description.values or []}
        values: list[Union[float, str, None]] = []
        for key in _column(load_table(attribute.values)):
            if key is None or key in missing:
                values.append(None)
            elif key in lookup:
                values.append(str(lookup[key]))
            else:
                logger.warning(f"Category attribute '{attribute.name}': key {key} is not in its lookup table.")
                values.append(None)
        return values
    return None


def _column(table: pa.Table, index: int = 0) -> list[Any]:
    values: list[Any] = table.column(index).to_pylist()
    return values
