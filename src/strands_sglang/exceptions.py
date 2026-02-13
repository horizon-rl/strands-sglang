# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom exceptions for SGLangClient.

These exceptions form the contract between client.py and sglang.py,
centralizing all error classification in the client layer so that
sglang.py never inspects raw HTTP status codes or response bodies.
"""


class SGLangClientError(Exception):
    """Base exception for all SGLangClient errors."""


class SGLangHTTPError(SGLangClientError):
    """HTTP error from SGLang server."""

    def __init__(self, message: str, *, status: int, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class SGLangContextLengthError(SGLangHTTPError):
    """Prompt/context exceeds the model's maximum length (400 + length keywords)."""


class SGLangThrottledError(SGLangHTTPError):
    """Rate-limited or temporarily unavailable (429, 503)."""


class SGLangConnectionError(SGLangClientError):
    """Connection-level failure (connect, timeout, DNS)."""


class SGLangDecodingError(SGLangClientError):
    """Server returned non-JSON response body."""
