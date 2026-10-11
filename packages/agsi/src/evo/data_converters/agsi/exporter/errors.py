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


class AGSIExporterException(Exception):
    """Base exception raised when exporting to AGSI fails."""


class UnsupportedObjectError(AGSIExporterException):
    """Raised when a Geoscience Object type cannot be exported to AGSI."""


class UnsupportedSectionGeometryError(UnsupportedObjectError):
    """Raised when section geometry cannot be written as AGSi top and bottom profiles.

    For example a section that is not a vertical plane, or a volume polygon whose boundary is not
    exactly one top and one bottom line over a common chainage range (overhangs, folds, holes).
    """
