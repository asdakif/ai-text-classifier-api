from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fastapi import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings


logger = logging.getLogger("ai_text_classifier_api")


@dataclass
class RateLimiter:
    max_requests: int
    window_seconds: int
    _requests: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def is_allowed(self, client_id: str, now: float) -> bool:
        window = self._requests[client_id]
        cutoff = now - self.window_seconds

        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            return False

        window.append(now)
        return True


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _get_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_request_middleware(
    settings: Settings,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        request.state.request_id = request_id

        if request.url.path == "/predict":
            is_allowed = rate_limiter.is_allowed(
                client_id=_get_client_id(request),
                now=time.time(),
            )
            if not is_allowed:
                response: Response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)

        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            process_time_ms,
        )
        return response

    return middleware
