# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from typing import Callable

from prometheus_async.aio import time as time_async_base
from prometheus_client import CollectorRegistry, Histogram, write_to_textfile

# All metrics are top-level singletons shared throughout the app

registry = CollectorRegistry()

# API Metrics
API_HISTOGRAM = Histogram(
    "request_latency_seconds",
    "API Request response times",
    ["service", "method"],
    registry=registry,
)

RC_MESSAGE_CONVERSION_HISTOGRAM = Histogram(
    "rc_message_conversion",
    "Time spend converting message to roadcast",
    ["converter"],
    registry=registry,
)


def dump_prometheus_metrics(to_file: str) -> None:
    write_to_textfile(to_file, registry)


def time_async(callable: Callable, name: str, module: str) -> Callable:
    return time_async_base(API_HISTOGRAM.labels(module, name))(callable)
