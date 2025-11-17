# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from __future__ import annotations

"""
Implements a Dispatcher type which manages a pool of available egodriver endpoints
and sensor simulation endpoints and assigns them to simulation rollouts as they
come available.
"""

import asyncio
import logging
import traceback
from dataclasses import dataclass

import alpasim_runtime
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import (
    NetworkSimulatorConfig,
    ScenarioConfig,
    UserEndpointConfig,
)
from alpasim_runtime.loop import UnboundRollout
from alpasim_runtime.scene_cache_monitor import SceneCacheMonitor
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.service_pool import ServicePool
from alpasim_runtime.services.traffic_service import TrafficService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Dispatcher:
    """
    Keeps track of contention of each microservice and assigns tasks as they come available
    """

    # there can be multiple service instances in the queue, depending on how many concurrent
    # tasks we want them to handle
    driver_pool: ServicePool[DriverService]
    sensorsim_pool: ServicePool[SensorsimService]
    physics_pool: ServicePool[PhysicsService]
    trafficsim_pool: ServicePool[TrafficService]
    controller_pool: ServicePool[ControllerService]

    camera_catalog: CameraCatalog
    sensorsim_scene_cache_monitor: SceneCacheMonitor

    do_shutdown: bool

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> list[str]:
        # Run all service compatibility checks in parallel
        results = await asyncio.gather(
            self.driver_pool.find_scenario_incompatibilities(scenario),
            self.sensorsim_pool.find_scenario_incompatibilities(scenario),
            self.physics_pool.find_scenario_incompatibilities(scenario),
            self.trafficsim_pool.find_scenario_incompatibilities(scenario),
            self.controller_pool.find_scenario_incompatibilities(scenario),
        )

        # Combine all results into a single list
        return [item for sublist in results for item in sublist]

    @staticmethod
    async def create(
        user_config: UserEndpointConfig,
        network_config: NetworkSimulatorConfig,
        camera_catalog: CameraCatalog,
        sensorsim_scene_cache_monitor: SceneCacheMonitor,
    ) -> Dispatcher:
        logger.info(
            f"Acquiring physics connection on {network_config.physics.addresses}..."
        )
        physics = await ServicePool.create(
            PhysicsService,
            user_config.physics,
            network_config.physics,
            connection_timeout_s=user_config.startup_timeout_s,
        )

        logger.info(
            f"Acquiring controller connection on {network_config.controller.addresses}..."
        )
        controller = await ServicePool.create(
            ControllerService,
            user_config.controller,
            network_config.controller,
            connection_timeout_s=user_config.startup_timeout_s,
        )

        logger.info(
            f"Acquiring traffic connection on {network_config.trafficsim.addresses}..."
        )
        traffic = await ServicePool.create(
            TrafficService,
            user_config.trafficsim,
            network_config.trafficsim,
            connection_timeout_s=user_config.startup_timeout_s,
        )

        logger.info(
            f"Acquiring sensorsim connection on {network_config.sensorsim.addresses}..."
        )
        sensorsim = await ServicePool.create(
            SensorsimService,
            user_config.sensorsim,
            network_config.sensorsim,
            connection_timeout_s=user_config.startup_timeout_s,
            camera_catalog=camera_catalog,
        )

        logger.info(
            f"Acquiring driver connection on {network_config.driver.addresses}..."
        )
        driver = await ServicePool.create(
            DriverService,
            user_config.driver,
            network_config.driver,
            connection_timeout_s=user_config.startup_timeout_s,
        )

        logger.info("Dispatcher ready.")
        return Dispatcher(
            driver_pool=driver,
            sensorsim_pool=sensorsim,
            physics_pool=physics,
            trafficsim_pool=traffic,
            controller_pool=controller,
            camera_catalog=camera_catalog,
            sensorsim_scene_cache_monitor=sensorsim_scene_cache_monitor,
            do_shutdown=user_config.do_shutdown,
        )

    def gather_version_ids(self) -> RolloutMetadata.VersionIds:
        runtime_version = alpasim_runtime.VERSION_MESSAGE

        versions = dict(
            nre=self.sensorsim_pool.get_version(),
            runtime=runtime_version,
            cosmos=self.driver_pool.get_version(),
            physics=self.physics_pool.get_version(),
            trafficsim=self.trafficsim_pool.get_version(),
            controller=self.controller_pool.get_version(),
        )

        for module, version in versions.items():
            logging.info(f"{module}: {version}")

        if not all(
            version.grpc_api_version == runtime_version.grpc_api_version
            for version in versions.values()
        ):
            logging.warning("API version mismatch.")

        return RolloutMetadata.VersionIds(
            runtime_version=runtime_version,
            egodriver_version=self.driver_pool.get_version(),
            sensorsim_version=self.sensorsim_pool.get_version(),
            physics_version=self.physics_pool.get_version(),
            traffic_version=self.trafficsim_pool.get_version(),
            controller_version=self.controller_pool.get_version(),
        )

    async def _run_rollout(
        self,
        rollout: UnboundRollout,
        driver: DriverService,
        sensorsim: SensorsimService,
        physics: PhysicsService,
        trafficsim: TrafficService,
        controller: ControllerService,
    ) -> bool:
        """
        Executes a task (session) using resources (driver and sensorsim instance) and then returns them to the pool.
        Returns `True` if the run was successful and `False` otherwise.
        """
        try:
            await rollout.bind(
                driver,
                sensorsim,
                physics,
                trafficsim,
                controller,
                self.camera_catalog,
            ).run()
            success = True
        except Exception as e:
            logger.error(
                f"Rollout {rollout.rollout_uuid} (scene {rollout.scene_id}) failed due to {e}.\n"
                f"{traceback.format_exc()}"
            )
            success = False

        # return the resources
        await self.driver_pool.put_back(driver)
        await self.sensorsim_pool.put_back(sensorsim)
        self.sensorsim_scene_cache_monitor.decrement(sensorsim, rollout.scene_id)
        await self.physics_pool.put_back(physics)
        await self.trafficsim_pool.put_back(trafficsim)
        await self.controller_pool.put_back(controller)

        return success

    async def dispatch_rollouts(self, rollouts: list[UnboundRollout]) -> bool:
        """
        Simulates all rollouts provided. Returns `True` if all of them exited successfully and
        `False` otherwise.
        """
        tasks = []
        for rollout in rollouts:
            driver = await self.driver_pool.get()
            sensorsim = await self.sensorsim_pool.get()
            self.sensorsim_scene_cache_monitor.increment(sensorsim, rollout.scene_id)
            physics = await self.physics_pool.get()
            trafficsim = await self.trafficsim_pool.get()
            controller = await self.controller_pool.get()
            logger.info(
                "Dispatching rollout %s (scene %s) on %s, %s, %s, %s, and %s.",
                rollout.rollout_uuid,
                rollout.scene_id,
                driver.name,
                sensorsim.name,
                physics.name,
                trafficsim.name,
                controller.name,
            )
            logger.info(
                "Number of still available services: \n\tDriver: %s\n\tSensorsim: %s\n"
                + "\tPhysics: %s\n\tTrafficsim: %s\n\tController: %s.",
                self.driver_pool.get_number_of_available_services(),
                self.sensorsim_pool.get_number_of_available_services(),
                self.physics_pool.get_number_of_available_services(),
                self.trafficsim_pool.get_number_of_available_services(),
                self.controller_pool.get_number_of_available_services(),
            )
            tasks.append(
                asyncio.create_task(
                    self._run_rollout(
                        rollout, driver, sensorsim, physics, trafficsim, controller
                    )
                )
            )
            logger.info(f"Current number of tasks: {len(tasks)}")

        # gather the tasks to await on the outstanding ones
        results = await asyncio.gather(*tasks)
        logger.info("Tasks finished")

        if self.do_shutdown:
            logger.info("Shutting down services.")
            await asyncio.gather(
                self.driver_pool.shut_down(),
                self.sensorsim_pool.shut_down(),
                self.trafficsim_pool.shut_down(),
                self.physics_pool.shut_down(),
                self.controller_pool.shut_down(),
            )
        else:
            logger.info("NOT shutting down services.")

        return all(results)
