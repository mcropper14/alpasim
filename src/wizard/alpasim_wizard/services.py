# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Service manager for organizing and building services."""

from __future__ import annotations

import itertools
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal, Optional

from .context import WizardContext
from .schema import ServiceConfig

logger = logging.getLogger(__name__)


@dataclass
class Address:
    host: str
    port: int

    def __repr__(self) -> str:
        return f"{self.host}:{self.port}"

    def is_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0


@dataclass
class VolumeMount:
    host: str
    container: str

    @staticmethod
    def from_str(string: str) -> VolumeMount:
        try:
            host, container = string.split(":")
        except ValueError as e:
            raise ValueError(f"Failed to convert {string=} to VolumeMount") from e
        return VolumeMount(host, container)

    def to_str(self) -> str:
        return f"{self.host}:{self.container}"

    def host_exists(self) -> bool:
        return os.path.exists(self.host)


@dataclass
class ContainerDefinition:
    """
    Contains the configuration of a specific container being deployed.
    """

    replica_idx: int
    name: str
    gpu: Optional[int]
    service_config: ServiceConfig
    context: WizardContext
    port: Optional[int]
    command: str
    workdir: Optional[str]
    environments: list[str]
    volumes: list[VolumeMount]
    address: Optional[Address]
    uuid: str

    @staticmethod
    def create(
        replica_idx: int,
        service_name: str,
        port: Optional[int],
        gpu: Optional[int],
        service_config: ServiceConfig,
        context: WizardContext,
        use_address_string: Literal["localhost", "0.0.0.0", "uuid"],
    ) -> ContainerDefinition:

        workdir = getattr(service_config, "workdir", None)
        environments = list(service_config.environments)
        volumes: list[VolumeMount] = []
        for volume_str in service_config.volumes:
            volumes.append(VolumeMount.from_str(volume_str))

        if getattr(context.cfg.wizard, "validate_mount_points", False):
            for volume in volumes:
                if not volume.host_exists():
                    raise OSError(f"{volume=} does not exist on host")

        uuid = f"{service_name}-{replica_idx}"

        return ContainerDefinition(
            replica_idx=replica_idx,
            name=service_name,
            gpu=gpu,
            service_config=service_config,
            context=context,
            port=port,
            command=ContainerDefinition._build_command(
                service_config, port, context, service_name
            ),
            workdir=workdir,
            environments=environments,
            volumes=volumes,
            address=ContainerDefinition._build_address(port, uuid, use_address_string),
            uuid=uuid,
        )

    @staticmethod
    def _build_command(
        service_config: ServiceConfig,
        port: Optional[int],
        context: WizardContext,
        service_name: str,
    ) -> str:
        # Build command with all replacements
        command = " ".join(service_config.command)

        assert (
            "{port}" not in command or port is not None
        ), f"Port is required for {service_name}"
        # Apply all variable replacements
        command = command.replace("{port}", str(port))
        command = command.replace("{sceneset}", context.sceneset_path or "None")

        # Runtime config name replacement
        runtime_config_name = f"generated-user-config-{int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))}.yaml"
        command = command.replace("{runtime_config_name}", runtime_config_name)

        return command

    @staticmethod
    def _build_address(
        port: Optional[int],
        uuid: str,
        use_address_string: Literal["localhost", "0.0.0.0", "uuid"],
    ) -> Optional[Address]:
        """Get the address of the container.

        Args:
            use_localhost: Whether to use localhost for the address. If False,
            the address will be the container UUID (for docker compose).

        Returns:
            The address of the container
        """
        if port is None:
            return None

        if use_address_string == "localhost":
            return Address(host="localhost", port=port)
        elif use_address_string == "0.0.0.0":
            return Address(host="0.0.0.0", port=port)
        elif use_address_string == "uuid":
            return Address(host=uuid, port=port)
        else:
            raise ValueError(f"Invalid address string: {use_address_string}")


@dataclass
class ContainerSet:
    """Container organization for deployment strategies."""

    sim: list = field(default_factory=list)
    eval: list = field(default_factory=list)
    agg: list = field(default_factory=list)
    runtime: list = field(default_factory=list)

    def all_containers(self) -> list:
        """Get all containers in execution order."""
        return self.sim + self.runtime + self.eval + self.agg


def create_gpu_assigner(gpu_ids: Optional[List[int]]) -> Iterator[Optional[int]]:
    """Create an iterator for GPU assignment."""

    def gpu_assigner() -> Iterator[Optional[int]]:
        if gpu_ids is None:
            yield from itertools.repeat(None)
        else:
            yield from itertools.cycle(gpu_ids)

    return gpu_assigner()


def build_container_set(
    context: WizardContext, use_address_string: Literal["localhost", "0.0.0.0", "uuid"]
) -> ContainerSet:
    """Build container set from configuration.

    Args:
        context: WizardContext containing configuration and state

    Returns:
        ContainerSet populated with containers for all configured services
    """
    cfg = context.cfg
    num_gpus = context.get_num_gpus()

    # Overwrite from config
    use_address_string = (
        "localhost"
        if context.cfg.wizard.debug_flags.use_localhost
        else use_address_string
    )

    def build_service_containers(
        service_name: str,
        service_config: ServiceConfig,
        runtime_cfg: Optional[Any] = None,
    ) -> List[ContainerDefinition]:
        """Build containers for a single service."""

        # Skip if not in services_to_run
        if service_name not in context.all_services_to_run:
            return []

        # Check if service should be skipped (skip: true in runtime config)
        if runtime_cfg:
            endpoints = getattr(runtime_cfg, "endpoints", {})
            service_endpoint = endpoints.get(service_name, {})
            if service_endpoint.get("skip", False):
                logger.debug(f"Skipping service {service_name} (marked as skip)")
                return []

        # Validate GPU configuration
        if (
            service_config.gpus is not None
            and num_gpus > 0
            and not all(gpu_id < num_gpus for gpu_id in service_config.gpus)
        ):
            raise RuntimeError(
                f"Service {service_name} requested GPUs {service_config.gpus} "
                f"but only 0 .. {num_gpus-1} are available."
            )

        # Build containers for all replicas
        gpu_assigner = create_gpu_assigner(service_config.gpus)
        containers = []

        for replica_idx, port, gpu in zip(
            range(service_config.replicas), context.port_assigner, gpu_assigner
        ):
            containers.append(
                ContainerDefinition.create(
                    replica_idx=replica_idx,
                    port=port,
                    gpu=gpu,
                    service_name=service_name,
                    service_config=service_config,
                    context=context,
                    use_address_string=use_address_string,
                )
            )

        return containers

    # Build containers for each service type
    sim_containers = []
    eval_containers = []
    agg_containers = []
    runtime_containers = []

    # Simulation services
    for name in cfg.wizard.run_sim_services or []:
        if name == "runtime":
            # Runtime handled separately
            runtime_containers = [
                ContainerDefinition.create(
                    service_name="runtime",
                    service_config=cfg.services.runtime,
                    replica_idx=0,
                    port=None,
                    gpu=None,
                    context=context,
                    use_address_string=use_address_string,
                )
            ]
        else:
            if config := getattr(cfg.services, name, None):
                sim_containers.extend(
                    build_service_containers(name, config, cfg.runtime)
                )

    # Evaluation services
    for name in cfg.wizard.run_eval_services or []:
        if config := getattr(cfg.services, name, None):
            eval_containers.extend(build_service_containers(name, config))

    # Aggregation services
    for name in cfg.wizard.run_aggregation_services or []:
        if config := getattr(cfg.services, name, None):
            agg_containers.extend(build_service_containers(name, config))

    logger.info("Built %d simulation containers", len(sim_containers))
    logger.info("Built %d evaluation containers", len(eval_containers))
    logger.info("Built %d aggregation containers", len(agg_containers))

    return ContainerSet(
        sim=sim_containers,
        eval=eval_containers,
        agg=agg_containers,
        runtime=runtime_containers,
    )
