# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.camera_catalog import CameraCatalog, CameraDefinition
from alpasim_runtime.config import (
    CameraDefinitionConfig,
    CameraIntrinsicsConfig,
    OpenCVPinholeConfig,
    PoseConfig,
    RuntimeCameraConfig,
)


def _make_local_camera_definition(logical_id: str) -> CameraDefinitionConfig:
    return CameraDefinitionConfig(
        logical_id=logical_id,
        rig_to_camera=PoseConfig(
            translation_m=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                focal_length=(800.0, 800.0),
                principal_point=(400.0, 200.0),
                radial=(0.0,) * 6,
                tangential=(0.0,) * 2,
                thin_prism=(0.0,) * 4,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )


@pytest.mark.asyncio
async def test_register_scene_appends_local_definition() -> None:
    local_cfg = _make_local_camera_definition("camera_local")
    catalog = CameraCatalog([local_cfg])

    sensorsim_camera_front = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_front",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_front",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )
    sensorsim_camera_local = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_local",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_local",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )

    sensorsim_cameras = [sensorsim_camera_front, sensorsim_camera_local]

    await catalog.merge_local_and_sensorsim_cameras("scene_a", sensorsim_cameras)

    merged = catalog.get_camera_definitions("scene_a")
    assert set(merged.keys()) == {"camera_front", "camera_local"}
    assert (
        merged["camera_local"].intrinsics.shutter_type
        == sensorsim_pb2.ShutterType.ROLLING_TOP_TO_BOTTOM
    )

    await catalog.merge_local_and_sensorsim_cameras("scene_a", sensorsim_cameras)
    refreshed = catalog.get_camera_definitions("scene_a")
    assert refreshed is not merged
    assert set(refreshed.keys()) == {"camera_front", "camera_local"}
    for key, original_def in merged.items():
        refreshed_def = refreshed[key]
        assert refreshed_def is not original_def
        assert refreshed_def.logical_id == original_def.logical_id
        assert (
            refreshed_def.intrinsics.SerializeToString()
            == original_def.intrinsics.SerializeToString()
        )
        assert (
            refreshed_def.rig_to_camera.as_grpc_pose().SerializeToString()
            == original_def.rig_to_camera.as_grpc_pose().SerializeToString()
        )


@pytest.mark.asyncio
async def test_register_scene_overwrites_intrinsics() -> None:
    local_cfg = _make_local_camera_definition("camera_front")
    catalog = CameraCatalog([local_cfg])

    sensorsim_camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_front",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_front",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )

    await catalog.merge_local_and_sensorsim_cameras("scene_b", [sensorsim_camera])

    merged = catalog.get_camera_definitions("scene_b")
    assert merged["camera_front"].intrinsics.shutter_type == (
        sensorsim_pb2.ShutterType.ROLLING_TOP_TO_BOTTOM
    )


@pytest.mark.asyncio
async def test_ensure_camera_defined_missing_definition_raises() -> None:
    catalog = CameraCatalog([])

    await catalog.merge_local_and_sensorsim_cameras("scene_a", [])

    scenario_camera = RuntimeCameraConfig(
        logical_id="camera_front",
        height=320,
        width=512,
        frame_interval_us=40_000,
        shutter_duration_us=20_000,
        first_frame_offset_us=1_000,
    )

    with pytest.raises(KeyError):
        catalog.ensure_camera_defined("scene_a", scenario_camera.logical_id)


def test_duplicate_local_definitions_raise() -> None:
    cfg = _make_local_camera_definition("camera_local")
    with pytest.raises(ValueError):
        CameraCatalog([cfg, cfg])


def test_pinhole_coeff_validations() -> None:
    base_kwargs = dict(
        focal_length=(800.0, 800.0),
        principal_point=(400.0, 200.0),
    )
    valid_radial = (0.0,) * 6
    valid_tangential = (0.0,) * 2
    valid_thin_prism = (0.0,) * 4

    bad_radial = CameraDefinitionConfig(
        logical_id="cam_bad_radial",
        rig_to_camera=PoseConfig(
            translation_m=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=(0.1,),
                tangential=valid_tangential,
                thin_prism=valid_thin_prism,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="radial must provide exactly 6"):
        CameraDefinition.from_config(bad_radial)

    bad_tangential = CameraDefinitionConfig(
        logical_id="cam_bad_tangential",
        rig_to_camera=bad_radial.rig_to_camera,
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=valid_radial,
                tangential=(0.1,),
                thin_prism=valid_thin_prism,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="tangential must provide exactly 2"):
        CameraDefinition.from_config(bad_tangential)

    bad_thin_prism = CameraDefinitionConfig(
        logical_id="cam_bad_thin_prism",
        rig_to_camera=bad_radial.rig_to_camera,
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=valid_radial,
                tangential=valid_tangential,
                thin_prism=(0.1,),
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="thin_prism must provide exactly 4"):
        CameraDefinition.from_config(bad_thin_prism)
