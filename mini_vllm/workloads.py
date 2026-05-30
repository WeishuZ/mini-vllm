"""Deterministic synthetic workloads for the engine and benchmarks."""
from __future__ import annotations

import random
from typing import List, Optional

from .request import Request


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _lognormal_len(rng: random.Random, mean: int, sigma: float, lo: int, hi: int) -> int:
    import math
    mu = math.log(max(1, mean))
    return _clamp(int(rng.lognormvariate(mu, sigma)), lo, hi)


def burst(
    n: int,
    prompt_mean: int = 256,
    gen_mean: int = 128,
    sigma: float = 0.5,
    seed: int = 0,
) -> List[Request]:
    """``n`` requests that all arrive at t=0 (saturated server)."""
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        p = _lognormal_len(rng, prompt_mean, sigma, 8, prompt_mean * 6)
        g = _lognormal_len(rng, gen_mean, sigma, 4, gen_mean * 6)
        reqs.append(Request(f"r{i}", prompt_len=p, max_tokens=g, arrival=0.0))
    return reqs


def poisson(
    n: int,
    rate_rps: float = 20.0,
    prompt_mean: int = 256,
    gen_mean: int = 128,
    sigma: float = 0.5,
    seed: int = 0,
) -> List[Request]:
    """``n`` requests with Poisson arrivals at ``rate_rps`` requests/second."""
    rng = random.Random(seed)
    reqs = []
    t_ms = 0.0
    for i in range(n):
        t_ms += rng.expovariate(rate_rps) * 1000.0
        p = _lognormal_len(rng, prompt_mean, sigma, 8, prompt_mean * 6)
        g = _lognormal_len(rng, gen_mean, sigma, 4, gen_mean * 6)
        reqs.append(Request(f"r{i}", prompt_len=p, max_tokens=g, arrival=t_ms))
    return reqs


def shared_prefix(
    n: int,
    system_len: int = 512,
    user_mean: int = 64,
    gen_mean: int = 96,
    sigma: float = 0.4,
    seed: int = 0,
) -> List[Request]:
    """``n`` requests that share an identical ``system_len``-token system prompt
    followed by a unique user message. Materializes ``token_ids`` so the prefix
    cache can dedup the shared blocks."""
    rng = random.Random(seed)
    system = list(range(1, system_len + 1))  # identical across requests
    reqs = []
    for i in range(n):
        u = _lognormal_len(rng, user_mean, sigma, 4, user_mean * 6)
        # unique user tokens drawn from a disjoint id range
        user = [system_len + 1 + rng.randrange(10_000) for _ in range(u)]
        token_ids = system + user
        g = _lognormal_len(rng, gen_mean, sigma, 4, gen_mean * 6)
        reqs.append(
            Request(
                f"r{i}",
                prompt_len=len(token_ids),
                max_tokens=g,
                arrival=0.0,
                token_ids=token_ids,
            )
        )
    return reqs


def fixed(
    n: int,
    prompt_len: int,
    gen_len: int,
    arrival_ms: float = 0.0,
    token_ids: Optional[List[int]] = None,
) -> List[Request]:
    """``n`` identical requests (handy for unit tests)."""
    return [
        Request(f"r{i}", prompt_len=prompt_len, max_tokens=gen_len,
                arrival=arrival_ms, token_ids=list(token_ids) if token_ids else None)
        for i in range(n)
    ]
