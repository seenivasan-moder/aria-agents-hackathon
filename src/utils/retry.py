"""Retry with exponential backoff — adapted from JARVIS Turbo."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.utils.http_pool import get_client


async def retry_request(
    method: str,
    url: str,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
) -> httpx.Response:
    client = await get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {"url": url}
            if json:
                kwargs["json"] = json
            if headers:
                kwargs["headers"] = headers
            if timeout:
                kwargs["timeout"] = timeout
            if method == "GET":
                r = await client.get(**kwargs)
            else:
                r = await client.post(**kwargs)
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

    raise last_error or ConnectionError(f"Request to {url} failed after {max_retries + 1} attempts")
