# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("alpasim_wizard")


@dataclass
class SceneIdAndUuid:
    scene_id: str
    uuid: str

    @staticmethod
    def list_from_df(df: pd.DataFrame) -> list[SceneIdAndUuid]:
        if "scene_id" not in df.columns or "uuid" not in df.columns:
            raise ValueError(
                f"DataFrame must have columns 'scene_id' and 'uuid'. Got {df.columns}."
            )

        return [
            SceneIdAndUuid(scene_id, uuid)
            for scene_id, uuid in zip(df["scene_id"], df["uuid"])
        ]
