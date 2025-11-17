# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging
import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.transforms as transforms
import numpy as np
import polars as pl
from tqdm import tqdm

from eval.aggregation.processing import ProcessedMetricDFs
from eval.data import EvaluationResultContainer, SimulationResult
from eval.schema import EvalConfig, MapElements
from eval.video_data import ShapelyMap

logger = logging.getLogger("alpasim.eval.video")

mpl.use("Agg")
mplstyle.use("fast")

VIDEO_FILE_NAME_FORMAT = "{clipgt_id}_{batch_id}_{rollout_id}.mp4"


def _compute_frame_timing(
    timestamps_us: np.ndarray,
    render_every_nth_frame: int,
) -> tuple[float, float]:
    """Derive animation interval (ms) and FPS from simulation timestamps."""
    if render_every_nth_frame < 1:
        raise ValueError("render_every_nth_frame must be at least 1")
    if len(timestamps_us) <= 1:
        raise ValueError("At least 2 timestamps are required")

    deltas_us = np.diff(timestamps_us.astype(np.int64))
    if not np.all(deltas_us == deltas_us[0]):
        logger.warning(
            "Timestamp deltas are not uniform: %s. Using median delta for frame timing.",
            deltas_us,
        )
    base_delta_us = float(np.median(deltas_us))
    frame_delta_us = base_delta_us * render_every_nth_frame

    fps = max(1e-6, 1_000_000.0 / frame_delta_us)
    interval_ms = frame_delta_us / 1_000.0
    return interval_ms, fps


def render_and_save_video_for_eval_container(
    evaluation_result_container: EvaluationResultContainer,
    processed_metric_dfs: ProcessedMetricDFs,
    output_dir: str,
    cfg: EvalConfig,
) -> None:
    logger.info(
        "Rendering video for evaluation container %s ",
        evaluation_result_container.file_path,
    )
    anim_1, fps = create_video_animation_for_eval_container(
        processed_metric_dfs,
        evaluation_result_container,
        cfg,
    )
    clipgt_id, batch_id, rollout_id = (
        evaluation_result_container.get_clipgt_batch_and_rollout_id()
    )
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    anim_1.save(
        os.path.join(
            output_dir,
            "videos",
            VIDEO_FILE_NAME_FORMAT.format(
                clipgt_id=clipgt_id, batch_id=batch_id, rollout_id=rollout_id
            ),
        ),
        fps=fps,
        dpi=100,
        writer="ffmpeg",
    )


def _setup_fig() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(9, 10))
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.03, hspace=0.03
    )

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1, 0.5],
        height_ratios=[1, 1],
    )
    axs = {}
    axs["map"] = fig.add_subplot(gs[0, 0])
    axs["table"] = fig.add_subplot(gs[0, 1])
    axs["image"] = fig.add_subplot(gs[1, 0:2])
    # axs["plans"] = fig.add_subplot(gs[1, 2])
    axs["map"].set_xticks([])
    axs["map"].set_yticks([])
    axs["table"].set_xticks([])
    axs["table"].set_yticks([])
    axs["image"].set_xticks([])
    axs["image"].set_yticks([])

    axs["map"].set_aspect("equal")
    # axs["plans"].set_aspect("equal")
    return fig, axs


def _list_in_dict_in_dict_to_list(
    artist_map: dict[str, dict[str, list[plt.Artist]]],
) -> list[plt.Artist]:
    all_artists = []
    for sub_dict in artist_map.values():
        for list_of_artists in sub_dict.values():
            all_artists.extend(list_of_artists)
    return all_artists


def get_ego_transform(
    sim_result: SimulationResult,
    cfg: EvalConfig,
    time: int,
) -> transforms.Affine2D:
    ego_transform = transforms.Affine2D()
    if cfg.video.map_video.rotate_map_to_ego:
        ego_yaw = (
            sim_result.actor_trajectories["EGO"]
            .interpolate_to_timestamps(np.array([time]))
            .poses[0]
            .yaw
        )
        ego_transform = ego_transform.rotate(np.pi / 2 - ego_yaw)

    return ego_transform


def render_table(
    ax: plt.Axes,
    processed_metric_dfs: ProcessedMetricDFs,
    evaluation_result_container: EvaluationResultContainer,
    time: int,
) -> mpl.table.Table:
    clipgt_id, batch_id, rollout_id = (
        evaluation_result_container.get_clipgt_batch_and_rollout_id()
    )

    run_name = processed_metric_dfs.trajectory_uid_df["run_name"][0]
    # Prepare aggregated data
    df_long_avg_t = (
        processed_metric_dfs.df_wide_avg_t.drop(
            "batch_id",
            "rollout_id",
            "clipgt_id",
            "run_name",
            "run_uuid",
            "trajectory_uid",
            "rollout_uid",
        )
        .unpivot()
        .sort("variable")
    )

    filtered_df_long = processed_metric_dfs.unprocessed_df.filter(
        pl.col("timestamps_us") == time,
    )

    assert (
        len(processed_metric_dfs.df_wide_avg_t) == 1
    ), f"Expected 1 row in df_wide_avg_t, got {len(processed_metric_dfs.df_wide_avg_t)}"
    assert (
        len(processed_metric_dfs.trajectory_uid_df) == 1
    ), f"Expected 1 row in trajectory_uid_df, got {len(processed_metric_dfs.trajectory_uid_df)}"
    # One row per metric
    assert len(filtered_df_long) == len(
        filtered_df_long["name"].unique()
    ), "Expected all metrics to be present in filtered_df_long"

    ax.axis("off")

    # Extract data from polars dataframe
    table_data = []
    # headers = ['Metric Name', 'Metric Value', 'Time Aggregation']
    col_names = ["Agg", "Per-Ts"]
    row_name = []

    for row in df_long_avg_t.iter_rows(named=True):
        row_name.append(row["variable"])
        curr_df = filtered_df_long.filter(
            pl.col("name") == row["variable"],
        )
        assert len(curr_df) <= 1
        # We might not have per-ts values for all ts.
        value_str = "N/A" if len(curr_df) == 0 else f"{curr_df['values'][0]:.2f}"
        # We also might not have a value for any ts.
        agg_value_str = "N/A" if row["value"] is None else f"{row['value']:.2f}"
        agg_str = processed_metric_dfs.agg_function_df.filter(
            pl.col("name") == row["variable"],
        )["time_aggregation"][0]
        table_data.append([f"{agg_value_str} ({agg_str})", value_str])

    table = ax.table(
        cellText=table_data,
        colLabels=col_names,
        rowLabels=row_name,
        loc="center right",
        cellLoc="center",
        rowLoc="left",
        edges="horizontal",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(col_names)):
        table[(0, i)].set_text_props(weight="bold")
        table[(0, i)].set_facecolor("#E6E6E6")
    # Style row labels
    for i in range(len(row_name)):
        table[(i + 1, -1)].set_text_props(weight="bold")
        table[(i + 1, -1)].set_facecolor("#E6E6E6")
    # Make table more narrow by adjusting column widths
    table.auto_set_column_width([0, 1])  # Reduce width multiplier for both columns

    # Add title with run name and clipgt id
    ax.text(
        0.0,
        1.0,
        f"Run: {run_name}\nClip: {clipgt_id}\nBatch: {batch_id}:{rollout_id}",
        ha="left",
        va="top",
        fontsize=6,
        transform=ax.transAxes,
    )

    # Remove top and bottom edges
    for col in range(len(col_names)):
        # Top row cells - remove top edge
        top_cell = table.get_celld()[(0, col)]
        top_cell.visible_edges = top_cell.visible_edges.replace("T", "")

    for col in range(len(col_names) + 1):
        # Bottom row cells - remove bottom edge
        bottom_cell = table.get_celld()[(len(row_name), col - 1)]
        bottom_cell.visible_edges = bottom_cell.visible_edges.replace("B", "")

    return table


def update_table(
    table: mpl.table.Table,
    processed_dfs: ProcessedMetricDFs,
    time: int,
) -> mpl.table.Table:
    celld = table.get_celld()
    # First row is header, column names are column -1!
    n_rows = max(map(lambda coords: coords[0], celld.keys())) + 1

    metric_names = [celld[(row, -1)].get_text().get_text() for row in range(1, n_rows)]

    # Want to update only the Value(ts) cells, which are column 1

    for row, metric_name in enumerate(metric_names, start=1):
        curr_df_long = processed_dfs.unprocessed_df.filter(
            pl.col("timestamps_us") == time,
            pl.col("name") == metric_name,
        )

        assert len(curr_df_long) <= 1
        value_str = (
            "N/A" if len(curr_df_long) == 0 else f"{curr_df_long['values'][0]:.2f}"
        )

        celld[(row, 1)].get_text().set_text(value_str)

    return table


def create_video_animation_for_eval_container(
    processed_metrics_dfs: ProcessedMetricDFs,
    evaluation_result_container: EvaluationResultContainer,
    cfg: EvalConfig,
) -> tuple[animation.FuncAnimation, float]:

    sim_result = evaluation_result_container.sim_result
    timestamps_us = sim_result.timestamps_us
    camera = sim_result.cameras.camera_by_logical_id[cfg.video.camera_id_to_render]
    shapely_map = ShapelyMap.from_vec_map(sim_result.vec_map)
    should_render_table = processed_metrics_dfs.df_wide_avg_t.shape[0] > 0

    fig, axs = _setup_fig()

    camera.render_image_at_time(timestamps_us[0], axs["image"])

    if should_render_table:
        table = render_table(
            axs["table"],
            processed_metrics_dfs,
            evaluation_result_container,
            timestamps_us[0],
        )

    text_artist = axs["table"].text(
        0.00,
        0.00,
        f"Timestamp_us: {timestamps_us[0]}",
        ha="left",
        va="bottom",
        transform=axs["table"].transAxes,
        fontsize=6,
    )

    ego_transform = get_ego_transform(
        sim_result=sim_result,
        cfg=cfg,
        time=timestamps_us[0],
    )

    image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
        axs["map"],
        "EGO",
        timestamps_us[0],
        cfg,
        axis_transform=ego_transform,
    )

    # Outer key: name of the element to plot
    # Inner key: name of the element artists to plot (e.g. border and fill)
    artists_on_map: dict[str, dict[str, list[plt.Artist]]] = {}

    artists_on_map["map"] = shapely_map.render(
        axs["map"],
        cfg,
        center=image_center_xy,
        max_dist=cfg.video.map_video.map_radius_m + 10,
    )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.GT_LINESTRING in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["gt_linestring"] = (
            sim_result.ego_recorded_ground_truth_trajectory.set_linestring_plot_style(
                "gt_linestring",
                linewidth=1,
                style="g-",
                alpha=0.7,
            ).render_linestring(axs["map"])
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
            axs["map"],
            timestamps_us[0],
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )
    else:
        artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
            axs["map"],
            timestamps_us[0],
            only_agents=["EGO"],
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["driver_responses"] = sim_result.driver_responses.render_at_time(
            axs["map"], timestamps_us[0], "now"
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["route"] = sim_result.routes.render_at_time(
            axs["map"],
            timestamps_us[0],
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.EGO_GT_GHOST_POLYGON in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["ego_gt_ghost_polygon"] = (
            sim_result.ego_recorded_ground_truth_trajectory.set_polygon_plot_style(
                fill_color="limegreen",
            ).render_polygon_at_time(axs["map"], timestamps_us[0])
        )

    for artist in _list_in_dict_in_dict_to_list(artists_on_map):
        artist.set_transform(ego_transform + axs["map"].transData)

    def update(time: int) -> list[plt.Artist]:
        if should_render_table:
            update_table(table, processed_metrics_dfs, time)
        camera_artist = camera.render_image_at_time(time, axs["image"])

        ego_transform = get_ego_transform(
            sim_result=sim_result,
            cfg=cfg,
            time=time,
        )
        image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
            axs["map"],
            "EGO",
            time,
            cfg,
            axis_transform=ego_transform,
        )

        artists_on_map["map"] = shapely_map.render(
            axs["map"],
            cfg,
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["driver_responses"] = (
                sim_result.driver_responses.render_at_time(axs["map"], time, "now")
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["route"] = sim_result.routes.render_at_time(
                axs["map"],
                time,
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.EGO_GT_GHOST_POLYGON
            in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["ego_gt_ghost_polygon"] = (
                sim_result.ego_recorded_ground_truth_trajectory.render_polygon_at_time(
                    axs["map"], time
                )
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
                axs["map"],
                time,
                center=image_center_xy,
                max_dist=cfg.video.map_video.map_radius_m + 10,
            )
        else:
            artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
                axs["map"],
                time,
                only_agents=["EGO"],
            )

        for artist in _list_in_dict_in_dict_to_list(artists_on_map):
            artist.set_transform(ego_transform + axs["map"].transData)

        text_artist.set_text(f"Time: {time}")

        all_artists = _list_in_dict_in_dict_to_list(artists_on_map)
        all_artists.append(camera_artist)
        if should_render_table:
            all_artists.append(table)
        all_artists.append(text_artist)
        return all_artists

    timestamps_to_render_us = timestamps_us[:: cfg.video.render_every_nth_frame]
    interval_ms, fps = _compute_frame_timing(
        timestamps_us, cfg.video.render_every_nth_frame
    )

    frames_iterator = (
        tqdm(timestamps_to_render_us, desc="Rendering animation frames")
        if cfg.num_processes == 1
        else timestamps_to_render_us
    )

    # Create animation with progress bar
    anim_1 = animation.FuncAnimation(
        fig,
        update,
        frames=frames_iterator,
        interval=interval_ms,
        blit=True,
    )

    return anim_1, fps
