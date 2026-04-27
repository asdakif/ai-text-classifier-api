from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected_api_key = request.app.state.settings.api_key
    if not expected_api_key:
        return

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
