"""Geometry and reduction checks for the MJLab-style foot height sensor."""

import math

import torch

from nest.sensors.foot_height import FootRingPatternCfg, foot_ring_pattern, reduce_foot_ray_heights


def test_foot_ring_has_center_and_six_vertical_rays():
    starts, directions = foot_ring_pattern(FootRingPatternCfg(), "cpu")
    assert starts.shape == directions.shape == (7, 3)
    torch.testing.assert_close(starts[0], torch.zeros(3))
    torch.testing.assert_close(torch.linalg.vector_norm(starts[1:, :2], dim=-1), torch.full((6,), 0.03))
    torch.testing.assert_close(starts[:, 2], torch.zeros(7))
    torch.testing.assert_close(directions, torch.tensor([[0.0, 0.0, -1.0]] * 7))
    assert math.isclose(starts[1, 0].item(), 0.03, abs_tol=1e-6)


def test_height_reduction_misses_and_backfaces():
    frame_z = torch.tensor([0.5, 2.0, 0.4, 0.3])
    hit_z = torch.tensor(
        [
            [0.2, float("inf"), 0.1],
            [float("inf"), float("inf"), float("inf")],
            [0.1, 0.2, float("inf")],
            [float("inf"), float("inf"), float("inf")],
        ]
    )
    normals = torch.tensor(
        [
            [1.0, float("inf"), 1.0],
            [float("inf"), float("inf"), float("inf")],
            [1.0, -1.0, float("inf")],
            [float("inf"), float("inf"), float("inf")],
        ]
    )
    actual = reduce_foot_ray_heights(frame_z, hit_z, 1.0, normals)
    torch.testing.assert_close(actual, torch.tensor([0.3, 1.0, 0.0, 0.3]))
