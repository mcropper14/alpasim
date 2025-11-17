# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging

from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper

logger = logging.getLogger("alpasim_wizard")
logger.setLevel(logging.INFO)


def check_config(cfg: AlpasimConfig) -> None:
    """
    Sanity-checks the config file. Can be used on the login node.
    """
    if cfg.services.sensorsim is None:
        # TODO: could we run in these conditions?
        raise ValueError("Missing 'sensorsim' config in 'services' section.")


def main() -> None:
    main_wrapper(check_config)


if __name__ == "__main__":
    main()
