<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://pypi.org/project/evo-data-converters-agsi/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-data-converters-agsi" /></a>
    <a href="https://github.com/SeequentEvo/evo-data-converters/actions/workflows/on-merge.yaml"><img src="https://github.com/SeequentEvo/evo-data-converters/actions/workflows/on-merge.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

## Evo

Evo is a unified platform for geoscience teams. It enables access, connection, computation, and management of subsurface data. This empowers better decision-making, simplified collaboration, and accelerated innovation. Evo is built on open APIs, allowing developers to build custom integrations and applications. Our open schemas, code examples, and SDK are available for the community to use and extend. 

Evo is powered by Seequent, a Bentley organisation.

## Pre-requisites

* Python virtual environment with Python 3.10, 3.11, or 3.12
* Git

## Installation

`pip install evo-data-converters-agsi`

## agsi converter

Converts [AGSi](https://ags-data-format-wg.gitlab.io/agsi/agsi_standard/latest/) ground models
to and from Evo `geological-sections` objects.

> **Status:** work in progress. This release contains the AGSi **reader**, the 2D **section
> geometry engine**, the **importer** (`convert_agsi`) and the **exporter** (`export_agsi`) for
> straight 2D sections. `export_blocksync_agsi` (block models) still raises `NotImplementedError`.

### Importing AGSi into Evo

`convert_agsi` converts each 2D section model of an AGSi 1.0.x package (a model with
`agsiGeometryAreaFromLines` elements) into one Evo `geological-sections` object
(`GeologicalSections_V1_2_0`) and publishes it, or returns it with `publish_objects=False`.

```python
from evo.data_converters.agsi import convert_agsi

objects = convert_agsi(
    "Silvertown2D_wkt_straight_synthetic.agsi",
    evo_workspace_metadata=workspace_metadata,  # or service_manager_widget=...
    coordinate_reference_system="EPSG:27700",
    base_elevation=-100.0,  # base of the lowest unit, which has a top profile only
    upload_path="agsi",
)
```

* **CRS.** The `globalXYSystem` declared in the AGSi file is never applied automatically, because
  it may be inferred rather than stated by the producer. Pass `coordinate_reference_system`; a
  warning is logged when it is missing or differs from the declared EPSG code. Coordinate system
  transformations (`transformShiftX`, ...) are not applied.
* **Elevations.** `base_elevation` / `top_elevation` close units without a bottom / top profile;
  they default to the model boundary's `bottomElevation` / `topElevation`.
* **Errors.** Curved alignments, 3D profiles and files without a 2D section model raise
  `AGSIUnsupportedGeometryError`; a missing required elevation or invalid data raise
  `AGSIInvalidDataError`. Nothing is published when an error is raised.

Object layout:

| Evo | From AGSi |
| --- | --- |
| `name`, the single `sections[0]` and `folders[0]` | `modelName` (else `modelID`); the folder is named after the alignment |
| `sections[0].origin`, `rotation` | the straight alignment at chainage 0 and elevation 0; vertical plane along the alignment |
| `line_geometry` | (chainage, elevation) vertices and chunks; each profile is stored once |
| `volumes` | one or more closed polygons per element, `feature` `OutputVolume`; `material_key` and `layer` are the `elementID`, `name` the `elementName` |
| `surfaces` | one per distinct top/bottom line (`ContactSurface`); generated horizontal lines are `BoundarySurface` |
| `materials`, `layer_order` | one per element (`colourRGB` or a default colour), top to bottom |
| `volume_attributes` | `elementID`, `elementType`, and one attribute per `agsiDataParameterValue` code (and case); numeric values are continuous, others strings |
| `extensions.agsi` | `schemaVersion`, `modelID`, `modelName`, `coordSystemID`, `alignmentID`, `alignmentName`, `startChainage`, `alignmentLength`; `parameterColumns` maps attribute names of parameters with a `caseID` to their `codeID` and `caseID` |

An element that pinches out (its top and bottom meet) is several volumes: an Evo volume is one
simple polygon and cannot hold separate areas or touch itself. Its volumes share the element's
`layer`, material and attributes. Which surfaces are the top and bottom of which element is not
recorded, so on export such an element may become several `"(part n)"` elements (see below).

Models without section areas (plan models holding alignments, 3D models of surfaces) are skipped.
`valueProfile` parameters are not converted.

### Exporting Evo to AGSi

`export_agsi` downloads Evo `geological-sections` objects and writes them to one AGSi 1.0.1
package: `index.json`, the main JSON file and one WKT file per distinct line.

```python
from evo.data_converters.agsi import export_agsi
from evo.data_converters.common import EvoObjectMetadata

export_agsi(
    "sections.agsi",
    objects=[EvoObjectMetadata(object_id=object_id)],  # optionally version_id=...
    evo_workspace_metadata=workspace_metadata,  # or service_manager_widget=...
)
```

`from evo.data_converters.agsi.exporter import build_agsi_export, SectionsSource` builds the same
package from objects already in memory: `build_agsi_export([SectionsSource(obj, load_table)], title)`
(`load_table` returns the `pyarrow.Table` of a data component); `.write(path)` writes it.

| AGSi | From Evo |
| --- | --- |
| `agsFile`, `agsProject` | the file name; per section model one `XZ` coordinate system (chainage, elevation; `systemID` from `coordSystemID`), and one `XY` system per distinct object CRS (`Evo-XY`; `globalXYSystem` `EPSG:n` for EPSG codes; name, axes and units from the CRS, `unknown` units without one) |
| section `agsiModel` | one per section; `modelID`, `modelName`, `coordSystemID` and `alignmentID` from `extensions.agsi` when the object has a single section, else derived from the object and section names; the Evo folder path is noted in `remarks` |
| plan `agsiModel`, `agsiModelAlignment` | one plan model per object holding a straight 2-point alignment through the section origin along its strike, from `startChainage` to `startChainage + alignmentLength` (from `extensions.agsi` when they cover the profiles, else the profiles' chainage range) |
| `agsiModelElement` | one per AGSi element (see below), in `layer_order`; `elementID` from the `elementID` attribute (else `layer`, `material_key`, `name`); `elementName` from the volume name when different; `colourRGB` from the material |
| `agsiGeometryAreaFromLines` | top and bottom WKT `LINESTRING (chainage elevation, ...)` profiles; identical lines (shared contacts) are written once and referenced by both units |
| `agsiModelBoundary` | `topElevation` / `bottomElevation` when a generated horizontal `BoundarySurface` line closes the units at one elevation; that line is then omitted from the profiles |
| `agsiDataParameterValue` | one per volume attribute value: continuous and integer → `valueNumeric`, string and category → `valueText`; `codeID`/`caseID` from `parameterColumns`, else the attribute name. Missing values are skipped; other attribute types (e.g. bool) are not exported (with a warning) |
| `agsProjectCodeSet` | one set per parameter attribute in use (`usedByObject` `agsiDataParameterValue`, `usedByAttribute` `codeID` and, when any are written, `caseID`), holding one `agsProjectCode` per distinct value written (exact string, first-use order, shared by all sections and objects). Evo keeps no code descriptions or units, so `description` repeats the code and `units` is omitted; none is written without parameters |

Constraints (each raises `UnsupportedSectionGeometryError` and nothing is written):

* The section must be a vertical plane (`dip` 90°); chainage is along strike and elevation is
  `v + origin.z`.
* Each volume must be a single ring without internal rings, with non-zero area, that splits at its
  lowest- and highest-chainage vertices into one top and one bottom line, each strictly increasing
  in chainage (no overhangs or vertical steps) with the top never below the bottom. Vertical edges
  at either end are left out of the lines. Where a polygon ends in a point, top and bottom meet
  there. No other split (such as halfway in elevation) is ever guessed.

Elements (top and bottom lines from the volumes of one element):

* Volumes are grouped by element ID. Volumes that meet end to end, exactly at single pinch-out
  vertices (and have the same name, description, material and attributes), are joined into one
  element. Otherwise each volume is written as its own element `"<elementID> (part n)"` with a
  remark and a warning; the zero-thickness runs between them are not invented, and a
  zero-thickness run at the end of an element is not written (its lines end where the volume
  ends). An element with zero thickness everywhere has no volume and is not written.
* `extensions.agsi.surfaces` (top/bottom relationships recorded by earlier versions of the
  importer) is ignored, so stale relationships never override the current volumes; it is not
  removed from the object.

Other behaviour:

* Sections without volumes are skipped; an object with no volumes, a non-`geological-sections`
  object or an empty object list raise `AGSIExporterException` / `UnsupportedObjectError`.
* Not kept: the original XY `systemID`, geometry descriptions, revisions and remarks, and the
  number formatting of the source WKT; duplicate source files of one contact become one file;
  palette colours become explicit `colourRGB`; mixed numeric/text parameters (stored as strings
  in Evo) come back as `valueText`; source code sets (code descriptions, units, `sourceURI`) are
  replaced by the generated sets above.

Round trip: the first export of an imported file writes each unit that pinches out (except at
exact single pinch-out vertices) as several `"(part n)"` elements, so the straight synthetic
Silvertown sample's 11 elements become 21 (Surficial 4 parts, Alluvium 5, River Terrace Deposits
2, Woolwich Lower Shelly Clay 2, Upnor 2), with types, descriptions and parameters of the source
element and every profile a vertex-identical run of a source line; contacts between parts stay
separate files (40 profiles). No geometry is invented. Importing that package again gives 22
volumes and 41 surfaces, and further round trips are stable: the second export only drops the
`"(part n)"` remarks (element remarks are not kept in Evo), after which the files are
byte-identical when the same title is used. The files are not byte-identical to the source (see
above). Exported packages validate against the AGSi 1.0.1 JSON schema, and every parameter
`codeID` and `caseID` is defined in a generated code set, so the official AGSi validator's code
reference check passes.

### Reading AGSi packages

`AgsiPackage.open` accepts:

* a `.agsi` (or `.zip`) archive with `index.json`, the main JSON file and its supporting files;
* a folder holding the extracted contents of such an archive;
* the main JSON file itself or `index.json`. Supporting files are resolved relative to its folder.

```python
from evo.data_converters.agsi.reader import AgsiPackage

package = AgsiPackage.open("Silvertown2D_wkt_1.agsi")
print(package.schema_version)  # "1.0.1"

section = package.models[0]
print(package.document.coordinate_system(section))  # agsProjectCoordinateSystem via coordSystemID

for model in package.models:
    for element in model.agsi_model_element:
        print(element.element_id, element.geometry_object_name)
        for role, geometry in element.geometry_files():  # role: "top", "bottom", ...
            data = package.read_geometry(geometry)
            if data.lines:
                vertices = data.single_line()  # (n, 2) chainage/elevation or (n, 3) x/y/z
            elif data.mesh is not None or data.polygons:
                mesh = data.to_triangle_mesh()  # vertices (n, 3), triangles (m, 3)
```

`read_agsi_file(filepath)` in `evo.data_converters.agsi.importer` returns the same `AgsiPackage`.
I/O problems raise `AGSIDataFileIOError`. Malformed documents raise `AGSIInvalidDataError`.

#### Supported AGSi versions

Only **AGSi 1.0.x** is supported, read as-is with no normalisation. The version is taken from
`agsSchema.version`, and any version declared in `index.json` must also be 1.0.x. Versions must be
written in full as `1.0.<patch>` (e.g. `1.0.0`, `1.0.1`); `1.0`, `1.0.`, `1.0.1.2`, prerelease
suffixes and surrounding whitespace are rejected. Other versions,
including the pre-1.0 AGSi v0.6 format (no `agsSchema`), raise `AGSIUnsupportedVersionError`
(a subclass of `AGSIInvalidDataError`).

Pydantic models (`reader.models`) cover the subset used by the converter: `agsSchema`, `agsFile`,
`agsProject` (incl. coordinate systems), `agsiModel`, `agsiModelElement`, `agsiModelBoundary`,
`agsiModelAlignment`, `agsiGeometryFromFile`, `agsiGeometryAreaFromLines`,
`agsiGeometryVolFromSurfaces`, `agsiGeometryPlane`, `agsiGeometryLayer` and
`agsiDataParameterValue`. Unknown attributes are kept (`model.extra_attributes`,
`model.to_agsi()`).

Following the AGSi
[model sections guidance](https://ags-data-format-wg.gitlab.io/agsi/AGSi_Documentation/Guidance_Model_Sections/),
a 2D section model gives its section line with `alignmentID`, which refers to an
`agsiModelAlignment` (usually in a separate plan model). Profile chainages are measured along
that alignment from its `startChainage`.

#### Geometry files

`reader.geometry_files.parse_geometry` returns numpy arrays (`GeometryData`) from:

* **WKT/EWKT**: `LINESTRING`, `POLYGON`, `MULTI*`, `GEOMETRYCOLLECTION` and TINs, 2D or 3D.
  Collections of triangles are returned as an indexed mesh.
* **DXF**: `LINE` chains, `LWPOLYLINE`, `POLYLINE` (incl. polyface meshes), `3DFACE` and `POINT`.
  Connected `LINE` entities (as in the Silvertown profiles) are joined into one polyline.
  `filePart` selects a layer.
* **STL**: ASCII and binary.

The reader tolerates quirks found in published examples:

* The declared `fileFormat` is checked against the file content; the content wins (logged).
* A `fileURI` that is not found exactly is matched case-insensitively (with a warning).
* Numbers split over two lines in Mathematica's "exponent above" style are repaired.

URIs that point outside the package are rejected, as is an `index.json` `filePath` that does
(absolute, drive, UNC or URL paths, and `..` segments that leave the folder holding `index.json`,
including percent-encoded ones). For extracted folders, the resolved path of each file read must
also stay inside the package folder, so symlinks and junctions cannot point outside it. DXF arcs (`ARC`, polyline bulges) and other
unsupported entities are not converted; they are counted in `GeometryData.curved_segments` and
`GeometryData.skipped_entities`.

### Section geometry

`evo.data_converters.agsi.geometry` turns a 2D section model (`agsiGeometryAreaFromLines`
elements) into section-plane geometry for Evo `geological-sections`; `convert_agsi` uses it.

```python
from evo.data_converters.agsi.geometry import build_model_section

section = build_model_section(package, package.models[0], base_elevation=-100.0)
frame, geometry = section.frame, section.geometry
frame.origin, frame.rotation           # world point at chainage 0, Evo rotation of the section plane
geometry.vertices, geometry.chunks     # (u, v) = (chainage, elevation) vertices; [offset, count] chunks
for volume in geometry.volumes:        # one or more closed polygons per element
    ring = geometry.ring(volume)
for line in geometry.lines:            # each top/bottom profile once, shared by the units above and below
    line.top_of, line.bottom_of, line.generated
frame.section_to_world(geometry.vertices)  # world x, y, z
```

* **Alignment.** The section frame comes from the `agsiModelAlignment` referenced by the
  model's `alignmentID`; without one, chainage is used as a local x axis (`SectionFrame.local()`),
  and you can pass `frame=` explicitly to override the placement. A linked alignment is still
  checked, so `frame=` cannot be used to accept a curved one. Only **straight** alignments are supported. Consecutive
  vertices or connected DXF `LINE`s are merged only when they are *exactly* collinear (exact
  arithmetic on the stored decimal values, no tolerance). Curved, kinked, closed, branched or
  disconnected alignments and DXF arcs raise `AGSIUnsupportedGeometryError` (or
  `AGSIInvalidDataError` for broken data). Nothing is smoothed or simplified.
* **Profiles** must be 2D (chainage, elevation). Duplicate vertices are dropped and profiles
  drawn right to left are reversed; vertical steps and overhangs (chainage not strictly
  increasing) are rejected. Profiles that run past the end of the alignment are only logged.
* **Units.** A unit's top and bottom must cover the same chainage range. Where they coincide
  (pinch-out) the unit is split into separate polygons; a unit with zero thickness everywhere has
  no polygon. A top below its bottom raises `AGSIInvalidDataError`, checked exactly. Pinch-out
  splitting needs top and bottom to share their chainage sampling. If they are sampled
  differently and touch strictly inside the unit, the polygon would touch itself, so
  `AGSIUnsupportedGeometryError` is raised. The exception is a generated horizontal top or base
  (below): it is sampled at the other line's chainages, which is exact.
* **Shared contacts.** Identical profiles (such as the base of one unit and the top of the next)
  are stored once and split at the same chainages, so neighbouring polygons share their
  boundaries exactly.
* **Missing top or bottom.** AGSi lets a unit have a top only (it extends down to the next
  surface). Here, a missing bottom becomes a horizontal line at `base_elevation`, or the model
  boundary's `bottomElevation`. A missing top uses `top_elevation` or `topElevation`. If neither
  is given, `AGSIInvalidDataError` is raised. Overlaps with lower units are not trimmed.
* DXF `LINE` end points are joined by the reader within 1e-9 before the exact checks.

### Tests

```shell
cd packages/common && uv run test-agsi
```

The committed fixtures under `tests/data/` are small AGSi 1.0.1 excerpts of the Silvertown
examples. The examples published on the AGSi website are AGSi v0.6, which is not supported. To
also run the integration tests (not committed; up to 80 MB), set `AGSI_SAMPLE_DIR` to a folder
containing AGSi 1.0.1 conversions of them, named like the originals (`Silvertown2D_{wkt,dxf}_{1,3}.agsi`
and `Silvertown3D_{wkt,dxf,stl}.agsi`). Their NB/SB alignments are curved, so the geometry and
importer tests check that they are rejected (the 3D models are rejected for their 3D profiles); an
optional `Silvertown2D_wkt_straight_synthetic.agsi` in the same folder (the same units on a
straight alignment) is built and converted end to end, then exported and checked against the
source profiles.

All tests are offline: the importer and exporter tests build and read the objects with a local
data client that writes parquet files to a temporary folder, mock the Evo clients, publishing and
downloads, and fail on any non-loopback network connection.


## Code of conduct

We rely on an open, friendly, inclusive environment. To help us ensure this remains possible, please familiarise yourself with our [code of conduct.](https://github.com/SeequentEvo/evo-data-converters/blob/main/CODE_OF_CONDUCT.md)

## License
Evo data converters are open source and licensed under the [Apache 2.0 license.](./LICENSE.md)

Copyright © 2026 Bentley Systems, Incorporated.

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and