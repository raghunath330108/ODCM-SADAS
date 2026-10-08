import unittest

from collect_nuscenes import (
    CLASS_TO_CATEGORY,
    allowed_image_box,
    meets_point_count_threshold,
    parse_args,
    sample_split_targets,
    solve_split_assignments,
)


class ImageBoxTests(unittest.TestCase):
    def test_regular_classes_must_fit_inside_image(self):
        self.assertIsNotNone(
            allowed_image_box("vehicle.car", (0, 0, 100, 100), 100, 100)
        )
        self.assertIsNone(
            allowed_image_box("vehicle.car", (-1, 0, 100, 100), 100, 100)
        )

    def test_every_class_must_fit_inside_image(self):
        for category in ("vehicle.truck", "vehicle.bus.rigid"):
            self.assertIsNone(
                allowed_image_box(category, (0, 0, 101, 100), 100, 100)
            )
            self.assertIsNone(
                allowed_image_box(category, (-1, 0, 100, 100), 100, 100)
            )


class PointCountTests(unittest.TestCase):
    def test_default_thresholds_accept_either_sensor(self):
        self.assertTrue(
            meets_point_count_threshold(
                {"num_lidar_pts": 6, "num_radar_pts": 0}, "car"
            )
        )
        self.assertTrue(
            meets_point_count_threshold(
                {"num_lidar_pts": 0, "num_radar_pts": 3}, "car"
            )
        )
        self.assertFalse(
            meets_point_count_threshold(
                {"num_lidar_pts": 5, "num_radar_pts": 2}, "car"
            )
        )

    def test_bicycle_has_lower_thresholds(self):
        self.assertTrue(
            meets_point_count_threshold(
                {"num_lidar_pts": 4, "num_radar_pts": 0}, "bicycle"
            )
        )
        self.assertTrue(
            meets_point_count_threshold(
                {"num_lidar_pts": 0, "num_radar_pts": 2}, "bicycle"
            )
        )
        self.assertFalse(
            meets_point_count_threshold(
                {"num_lidar_pts": 3, "num_radar_pts": 1}, "bicycle"
            )
        )


class SplitTests(unittest.TestCase):
    def test_requested_250_sample_split_rounding(self):
        self.assertEqual(
            sample_split_targets(250),
            {"train": 175, "val": 38, "test": 37},
        )

    def test_larger_quota_split_rounding_and_cli(self):
        self.assertEqual(
            sample_split_targets(450),
            {"train": 315, "val": 68, "test": 67},
        )
        args = parse_args(["--dataset-root", "/dataset", "--quota-per-class", "450"])
        self.assertEqual(args.quota_per_class, 450)

    def test_multi_class_frames_can_satisfy_multiple_quotas(self):
        candidates = {
            "frame-ab": {"car", "truck"},
            "frame-a": {"car"},
            "frame-b": {"truck"},
        }
        targets = {name: 1 for name in CLASS_TO_CATEGORY}
        targets["car"] = 1
        targets["truck"] = 1
        targets.update({name: 0 for name in CLASS_TO_CATEGORY if name not in {"car", "truck"}})

        assignments, split_targets = solve_split_assignments(
            candidates, targets, seed=42
        )
        self.assertEqual(len(assignments), 1)
        self.assertEqual(set(assignments), {"frame-ab"})
        self.assertEqual(split_targets["car"]["train"], 1)
        self.assertEqual(split_targets["truck"]["train"], 1)

    def test_multi_class_overlap_does_not_exceed_exact_class_quotas(self):
        candidates = {
            "frame-car-truck": {"car", "truck"},
            "frame-car-person": {"car", "person"},
            "frame-truck": {"truck"},
            "frame-person": {"person"},
        }
        targets = {name: 0 for name in CLASS_TO_CATEGORY}
        targets.update({"car": 1, "truck": 1, "person": 1})

        assignments, _ = solve_split_assignments(candidates, targets, seed=42)

        for class_name in ("car", "truck", "person"):
            memberships = sum(
                class_name in candidates[token] for token in assignments
            )
            self.assertEqual(memberships, 1)


if __name__ == "__main__":
    unittest.main()
