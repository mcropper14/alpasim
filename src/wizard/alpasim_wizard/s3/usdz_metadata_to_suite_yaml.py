# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
This script takes a usdz_metadata.csv file produced by the populate_db script
and creates a suite_yaml file that can be input to the create_suite script.
The output yaml file has the following format:

```yaml
suites:
    - name: <name_of_the_suite>
    scenes:
        - <scene_id1>
        - <scene_id2>
```
"""

import argparse
import logging

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpasim_wizard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz-metadata-csv", type=str, required=True)
    parser.add_argument("--suite-name", type=str, required=True)
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.usdz_metadata_csv)
    config = {
        "suites": [
            {
                "name": args.suite_name,
                "scenes": metadata_df["scene_id"].tolist(),
            }
        ]
    }

    suite_yaml_path = f"suite_{args.suite_name}.yaml"
    with open(suite_yaml_path, "w") as f:
        yaml.dump(config, f)

    # Print the yaml file
    with open(suite_yaml_path, "r") as f:
        logger.info(f"{f.read()}")

    logger.info(f"Suite yaml file created at {suite_yaml_path}")
    logger.info(f"Total number of scenes: {len(metadata_df)}")

    # Unique artifact versions
    unique_artifact_versions = metadata_df["nre_version_string"].unique()
    logger.info(f"Unique artifact versions: {unique_artifact_versions}")


if __name__ == "__main__":
    main()
