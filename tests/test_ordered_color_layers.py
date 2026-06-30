import click
import numpy as np
import pytest
import torch

from bananaforge.cli import (
    _build_ordered_color_layers,
    _map_ordered_colors_to_materials,
    _parse_hex_color_order,
)


def test_parse_hex_color_order_requires_valid_distinct_hex_values():
    assert _parse_hex_color_order("#000000,#FFFFFF,#FFD700") == [
        (0, 0, 0),
        (255, 255, 255),
        (255, 215, 0),
    ]

    with pytest.raises(click.ClickException):
        _parse_hex_color_order("#000000")

    with pytest.raises(click.ClickException):
        _parse_hex_color_order("#000000,#000000")

    with pytest.raises(click.ClickException):
        _parse_hex_color_order("#000000,FFD700")


def test_map_ordered_colors_to_distinct_nearest_materials():
    selected_colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 215 / 255, 0.0],
        ]
    )

    material_indices = _map_ordered_colors_to_materials(
        [(0, 0, 0), (255, 255, 255), (255, 215, 0)],
        selected_colors,
        ["black", "white", "yellow"],
    )

    assert material_indices == [0, 1, 2]


def test_map_ordered_colors_rejects_duplicate_material_mapping():
    selected_colors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    with pytest.raises(click.ClickException):
        _map_ordered_colors_to_materials(
            [(1, 1, 1), (2, 2, 2)],
            selected_colors,
            ["black", "white"],
        )


def test_build_ordered_color_layers_creates_cumulative_material_masks():
    target_image = np.array(
        [[[0, 0, 0], [255, 255, 255], [255, 215, 0]]],
        dtype=np.uint8,
    )

    height_map, assignments = _build_ordered_color_layers(
        target_image_np=target_image,
        ordered_colors_rgb=[(0, 0, 0), (255, 255, 255), (255, 215, 0)],
        ordered_material_indices=[0, 1, 2],
        color_layer_count=2,
        device="cpu",
    )

    assert height_map.shape == (1, 1, 1, 3)
    assert height_map.squeeze().tolist() == [2.0, 4.0, 6.0]
    assert assignments.shape == (6, 1, 3)
    assert assignments.tolist() == [
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[-1, 1, 1]],
        [[-1, 1, 1]],
        [[-1, -1, 2]],
        [[-1, -1, 2]],
    ]
