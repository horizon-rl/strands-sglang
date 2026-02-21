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

"""Unit tests for cached utility functions."""

from unittest.mock import MagicMock

from strands_sglang.utils import get_client, get_client_from_slime_args


class TestGetClient:
    """Tests for get_client caching and get_client_from_slime_args."""

    def setup_method(self):
        get_client.cache_clear()

    def test_same_args_return_same_instance(self):
        """Same arguments return the exact same client object."""
        client1 = get_client("http://localhost:30000", max_connections=100)
        client2 = get_client("http://localhost:30000", max_connections=100)
        assert client1 is client2

    def test_different_args_return_different_instance(self):
        """Different arguments return different client objects."""
        client1 = get_client("http://localhost:30000", max_connections=100)
        client2 = get_client("http://localhost:30001", max_connections=100)
        assert client1 is not client2

    def test_slime_args_builds_url_and_max_connections(self):
        """get_client_from_slime_args computes URL and max_connections from slime args."""
        args = MagicMock()
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 9000
        args.sglang_server_concurrency = 256
        args.rollout_num_gpus = 8
        args.rollout_num_gpus_per_engine = 2

        client = get_client_from_slime_args(args)

        assert client.base_url == "http://10.0.0.1:9000"
        assert client._max_connections == 1024  # 256 * 8 // 2
