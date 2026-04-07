"""
Fetch running/loaded model status from OpenAI-compatible backends
that expose a running-models endpoint (e.g. llama-swap /running).

Expected response format from the running models URL:
{
  "running": [
    {"model": "model-id", "state": "ready", "ttl": 900, ...}
  ]
}

Models with state "ready" and a positive ttl are considered loaded.
The ttl (seconds until unload) is converted to an expires_at unix timestamp.
"""

import asyncio
import logging
import time

import aiohttp

log = logging.getLogger(__name__)


async def fetch_running_models(url: str, timeout: int = 3) -> dict:
    """
    GET a single running-models URL and return {model_id: expires_at}.
    Returns empty dict on any failure.
    """
    try:
        _timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=_timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                data = await response.json()

        result = {}
        for entry in data.get('running', []):
            model_id = entry.get('model', '')
            ttl = entry.get('ttl', 0)
            state = entry.get('state', '')
            if model_id and state == 'ready' and ttl > 0:
                result[model_id] = int(time.time()) + ttl
        return result
    except Exception as e:
        log.debug(f'Failed to fetch running models from {url}: {e}')
        return {}


async def fetch_all_openai_running_models(
    configs: dict, base_urls: list
) -> dict:
    """
    Check all OpenAI connections that have a running_model_url configured.
    Returns a merged {model_id: expires_at} dict with prefix_id applied.
    """
    tasks = []
    task_meta = []  # parallel list tracking prefix_id per task

    for idx, _url in enumerate(base_urls):
        config = configs.get(str(idx), configs.get(_url, {}))
        running_url = config.get('running_model_url', '')
        if not running_url:
            continue
        if not config.get('enable', True):
            continue

        prefix_id = config.get('prefix_id', '')
        tasks.append(fetch_running_models(running_url))
        task_meta.append(prefix_id)

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks)

    merged = {}
    for prefix_id, running in zip(task_meta, results):
        for model_id, expires_at in running.items():
            # Apply prefix_id to match Open WebUI's namespaced model IDs
            key = f'{prefix_id}.{model_id}' if prefix_id else model_id
            merged[key] = expires_at

    return merged
