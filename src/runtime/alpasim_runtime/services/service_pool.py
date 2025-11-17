# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Service pool implementation for service architecture.
"""

from __future__ import annotations

import asyncio
from asyncio import Queue
from typing import Any, Generic, List, Type, TypeVar

from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_runtime.config import (
    EndpointAddresses,
    ScenarioConfig,
    SingleUserEndpointConfig,
)
from alpasim_runtime.services.service_base import ServiceBase

ServiceType = TypeVar("ServiceType", bound=ServiceBase)

# Number of skip services to create in skip mode.
# This should be at least as large as nr_concurrent_rollouts * nr_replicas for
# all of the other services.
NR_SKIP_SERVICES = 100


class ServicePool(Generic[ServiceType]):
    """
    Generic service pool that works with any ServiceBase-derived service.
    """

    def __init__(
        self,
        services: List[ServiceType],
        version: VersionId,
    ):
        self.queue: Queue[ServiceType] = Queue()
        self.services = services
        self.version = version

        # Add all services to the queue
        for service in services:
            self.queue.put_nowait(service)

    @classmethod
    async def create(
        cls,
        service_class: Type[ServiceType],
        user_config: SingleUserEndpointConfig,
        addresses: EndpointAddresses,
        connection_timeout_s: int,
        **service_kwargs: Any,
    ) -> ServicePool[ServiceType]:
        """Create a service pool for the given service type."""
        services: List[ServiceType] = []
        versions: List[VersionId] = []

        if user_config.skip:
            # Skip mode: create single skip service
            for i in range(NR_SKIP_SERVICES):
                service = service_class(
                    "skip",
                    skip=True,
                    connection_timeout_s=connection_timeout_s,
                    id=i,
                    **service_kwargs,
                )
                services.append(service)
            versions.append(await service.get_version())
        else:
            # Real mode: create services for each address/replica
            for address in addresses.addresses:
                for i in range(user_config.n_concurrent_rollouts):
                    service = service_class(
                        address,
                        skip=False,
                        connection_timeout_s=connection_timeout_s,
                        id=i,
                        **service_kwargs,
                    )
                    services.append(service)

                # Get version from one instance per address
                version = await services[-1].get_version()
                versions.append(version)

        # Verify all versions match
        if not all(v.version_id == versions[0].version_id for v in versions):
            raise AssertionError(
                f"Incoherent versions: {[v.version_id for v in versions]}"
            )

        pool = cls(services, versions[0])

        return pool

    async def get(self) -> ServiceType:
        """Get a service instance from the pool."""
        return await self.queue.get()

    async def put_back(self, service: ServiceType) -> None:
        """Return a service instance to the pool."""
        await self.queue.put(service)

    def get_version(self) -> VersionId:
        """Get the version of services in this pool."""
        return self.version

    def get_number_of_services(self) -> int:
        """Get the number of services in this pool."""
        return len(self.services)

    def get_number_of_available_services(self) -> int:
        """Get the number of services in the queue."""
        return self.queue.qsize()

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> List[str]:
        """Check if services in this pool can handle the scenario."""
        if self.services:
            return await self.services[0].find_scenario_incompatibilities(scenario)
        return []

    async def shut_down(self) -> None:
        """Shutdown all services in the pool."""
        # Deduplicate services by address to avoid shutting down the same service multiple times
        seen_addresses = set()
        unique_services = []
        for service in self.services:
            if service.address not in seen_addresses:
                seen_addresses.add(service.address)
                unique_services.append(service)

        await asyncio.gather(*[service.shut_down() for service in unique_services])
