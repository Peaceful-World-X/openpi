import asyncio
from collections.abc import Mapping
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.perf_counter()
                request = msgpack_numpy.unpackb(await websocket.recv())
                obs, rtc_context = _decode_request(request)

                infer_start = time.perf_counter()
                if rtc_context is None:
                    action = self._policy.infer(obs)
                else:
                    action = self._policy.infer(
                        obs,
                        inference_delay=rtc_context["inference_delay"],
                        prev_chunk_left_over=rtc_context["prev_chunk_left_over"],
                    )
                infer_ms = (time.perf_counter() - infer_start) * 1000

                client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
                policy_infer_ms = action.get("policy_timing", {}).get("infer_ms")
                if policy_infer_ms is not None:
                    logger.info("[%s] Infer: %.2f ms (server) %.2f ms (model)", client_ip, infer_ms, policy_infer_ms)
                else:
                    logger.info("[%s] Infer: %.2f ms", client_ip, infer_ms)

                action["server_timing"] = {
                    "infer_ms": infer_ms,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.perf_counter() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _decode_request(request: object) -> tuple[dict, dict | None]:
    """Decode the flat legacy or nested stateless RTC request format."""
    if not isinstance(request, Mapping):
        raise ValueError(f"policy request must be a mapping, got {type(request).__name__}")

    has_context = "rtc_context" in request
    has_observation = "observation" in request
    if not has_context:
        # Preserve the original protocol for arbitrary flat observation
        # dictionaries, including one that happens to use an ``observation``
        # key for its own data contract.  The nested RTC envelope is selected
        # only when its context field is present as well.
        return dict(request), None

    if not has_observation:
        raise ValueError("RTC request must include an observation mapping")
    observation = request["observation"]
    context = request["rtc_context"]
    if not isinstance(observation, Mapping):
        raise ValueError("request.observation must be a mapping")
    if not isinstance(context, Mapping):
        raise ValueError("request.rtc_context must be a mapping")
    required = ("inference_delay", "prev_chunk_left_over")
    missing = [key for key in required if key not in context]
    if missing:
        raise ValueError(f"request.rtc_context missing required fields: {', '.join(missing)}")

    delay = np.asarray(context["inference_delay"])
    if delay.size != 1 or not np.issubdtype(delay.dtype, np.integer):
        raise ValueError("rtc_context.inference_delay must be a scalar integer")
    prefix = np.asarray(context["prev_chunk_left_over"])
    if prefix.ndim != 2:
        raise ValueError("rtc_context.prev_chunk_left_over must be a 2D action sequence")
    return dict(observation), {
        "inference_delay": int(delay.item()),
        "prev_chunk_left_over": prefix,
    }


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
