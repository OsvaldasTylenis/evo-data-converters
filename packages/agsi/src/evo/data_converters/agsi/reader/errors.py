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


class AGSIInvalidDataError(Exception):
    """Raised when the contents of an AGSi file are malformed or fail validation."""


class AGSIDataFileIOError(Exception):
    """Raised when an AGSi file, package, or supporting file cannot be opened or read."""


class AGSIUnsupportedVersionError(AGSIInvalidDataError):
    """Raised when an AGSi document is not AGSi 1.0.x (e.g. a pre-1.0 v0.6 document)."""
