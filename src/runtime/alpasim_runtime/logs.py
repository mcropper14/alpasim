# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Self

from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.metrics import time_async
from alpasim_utils.logs import LogWriter

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LogWriterManager:
    """Composes multiple LogWriter instances and decorates them with profiling info"""

    log_writers: list[LogWriter] = field(default_factory=list)

    async def log_message(self, message: LogEntry) -> None:

        log_tasks = [
            time_async(log_writer.log_message, name="log_message", module="logging")(
                message
            )
            for log_writer in self.log_writers
        ]

        await asyncio.gather(*log_tasks)

    async def __aenter__(self) -> Self:
        aenter_tasks = [log_writer.__aenter__() for log_writer in self.log_writers]
        await asyncio.gather(*aenter_tasks)
        return self

    async def __aexit__(self, *args, **kwargs) -> None:
        aexit_tasks = [
            log_writer.__aexit__(*args, **kwargs) for log_writer in self.log_writers
        ]
        await asyncio.gather(*aexit_tasks)
