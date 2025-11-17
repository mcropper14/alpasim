# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Driver service implementation."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type, TypeAlias

import numpy as np
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruth,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import EgodriverServiceStub
from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn
from alpasim_runtime.logs import LogEntry, LogWriter
from alpasim_runtime.services.service_base import (
    WILDCARD_SCENE_ID,
    ServiceBase,
    SessionInfo,
    timed_method,
)
from alpasim_runtime.types import ImageWithMetadata
from alpasim_utils.polyline import Polyline
from alpasim_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)

AvailableCamera: TypeAlias = AvailableCamerasReturn.AvailableCamera


class DriverService(ServiceBase[EgodriverServiceStub]):
    """
    Service for interacting with the autonomous driving policy.

    This service handles communication with the driver model, including
    submitting sensor observations and receiving driving decisions.
    """

    @property
    def stub_class(self) -> Type[EgodriverServiceStub]:
        return EgodriverServiceStub

    # Override the session method to add typed parameters
    def session(  # type: ignore[override]
        self,
        uuid: str,
        log_writer: LogWriter,
        random_seed: int,
        sensorsim_cameras: list[AvailableCamera],
        scene_id: Optional[str] = None,
    ) -> "ServiceBase[EgodriverServiceStub]":
        """Create a driver session with typed parameters.

        These are used in `_initialize_session()`.
        """
        return super().session(
            uuid=uuid,
            log_writer=log_writer,
            random_seed=random_seed,
            sensorsim_cameras=sensorsim_cameras,
            scene_id=scene_id,
        )

    @timed_method("initialize_session")
    async def _initialize_session(
        self, session_info: SessionInfo, **kwargs: Any
    ) -> None:
        """Initialize driver session after gRPC connection is established."""
        await super()._initialize_session(session_info=session_info)

        random_seed: int = session_info.additional_args["random_seed"]
        scene_id: str | None = session_info.additional_args.get("scene_id")
        sensorsim_cameras: list[AvailableCamera] = session_info.additional_args[
            "sensorsim_cameras"
        ]

        rollout_spec = DriveSessionRequest.RolloutSpec(
            vehicle=DriveSessionRequest.RolloutSpec.VehicleDefinition(
                available_cameras=sensorsim_cameras,
            ),
        )

        request = DriveSessionRequest(
            session_uuid=self.session_info.uuid,
            random_seed=random_seed,
            debug_info=DriveSessionRequest.DebugInfo(scene_id=scene_id),
            rollout_spec=rollout_spec,
        )

        await self.session_info.log_writer.log_message(
            LogEntry(driver_session_request=request)
        )

        if self.skip:
            return

        await self.stub.start_session(request)

    @timed_method("cleanup_session")
    async def _cleanup_session(self, **kwargs: Any) -> None:
        """Clean up driver session."""
        if self.skip:
            return

        close_request = DriveSessionCloseRequest(session_uuid=self.session_info.uuid)
        await self.stub.close_session(close_request)

    async def get_available_scenes(self) -> List[str]:
        """Get list of available scenes from the driver service."""
        return [WILDCARD_SCENE_ID]

    @timed_method("submit_image")
    async def submit_image(self, image: ImageWithMetadata) -> None:
        """Submit an image observation for the current session."""
        request = RolloutCameraImage(
            session_uuid=self.session_info.uuid,
            camera_image=RolloutCameraImage.CameraImage(
                frame_start_us=image.start_timestamp_us,
                frame_end_us=image.end_timestamp_us,
                image_bytes=image.image_bytes,
                logical_id=image.camera_logical_id,
            ),
        )

        await self.session_info.log_writer.log_message(
            LogEntry(driver_camera_image=request)
        )

        if self.skip:
            return

        await self.stub.submit_image_observation(request)

    @timed_method("submit_trajectory")
    async def submit_trajectory(self, trajectory: Trajectory) -> None:
        """Submit an egomotion trajectory for the current session."""
        request = RolloutEgoTrajectory(
            session_uuid=self.session_info.uuid,
            trajectory=trajectory.to_grpc(),
        )

        await self.session_info.log_writer.log_message(
            LogEntry(driver_ego_trajectory=request)
        )

        if self.skip:
            return

        await self.stub.submit_egomotion_observation(request)

    @timed_method("submit_route")
    async def submit_route(
        self, timestamp_us: int, route_polyline_in_rig: Polyline
    ) -> None:
        """Submit a route for the current session."""
        # Convert the route polyline to gRPC Route format
        grpc_route = route_polyline_in_rig.to_grpc_route(timestamp_us)

        request = RouteRequest(
            session_uuid=self.session_info.uuid,
            route=grpc_route,
        )

        await self.session_info.log_writer.log_message(LogEntry(route_request=request))

        if self.skip:
            return

        await self.stub.submit_route(request)

    @timed_method("submit_recording_ground_truth")
    async def submit_recording_ground_truth(
        self, timestamp_us: int, trajectory: Trajectory
    ) -> None:
        """Submit ground truth from recording for the current session."""
        request = GroundTruthRequest(
            session_uuid=self.session_info.uuid,
            ground_truth=GroundTruth(
                timestamp_us=timestamp_us,
                trajectory=trajectory.to_grpc(),
            ),
        )

        await self.session_info.log_writer.log_message(
            LogEntry(ground_truth_request=request)
        )

        if self.skip:
            return

        await self.stub.submit_recording_ground_truth(request)

    @timed_method("drive")
    async def drive(
        self, time_now_us: int, time_query_us: int, renderer_data: Optional[bytes]
    ) -> Trajectory:
        """Request a drive decision for the current session."""
        # Create request with both old and new fields for backward compatibility
        request = DriveRequest(
            session_uuid=self.session_info.uuid,
            time_now_us=time_now_us,
            time_query_us=time_query_us,
            renderer_data=renderer_data or b"",
        )

        await self.session_info.log_writer.log_message(LogEntry(driver_request=request))

        if self.skip:
            # Create a simple trajectory response
            from alpasim_utils.qvec import QVec

            # Create QVec poses for the trajectory
            poses = QVec.stack(
                [
                    QVec(
                        vec3=np.array([0.0, 0.0, 0.0]),
                        quat=np.array([0.0, 0.0, 0.0, 1.0]),
                    ),
                    QVec(
                        vec3=np.array([0.0, 0.0, 0.0]),
                        quat=np.array([0.0, 0.0, 0.0, 1.0]),
                    ),
                ]
            )

            trajectory = Trajectory(
                timestamps_us=np.array([time_now_us, time_query_us], dtype=np.uint64),
                poses=poses,
            )
            response = DriveResponse(
                trajectory=trajectory.to_grpc(),
            )
        else:
            response = await self.stub.drive(request)

        await self.session_info.log_writer.log_message(LogEntry(driver_return=response))

        return Trajectory.from_grpc(response.trajectory)
