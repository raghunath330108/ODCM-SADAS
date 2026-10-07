import unittest

from collect_nuscenes import (
    CLASS_TO_CATEGORY,
    allowed_image_box,
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

    def test_truck_and_bus_may_have_ten_percent_outside(self):
        self.assertIsNotNone(
            allowed_image_box("vehicle.truck", (0, 0, 110, 100), 100, 100)
        )
        self.assertIsNotNone(
            allowed_image_box("vehicle.bus.rigid", (-10, 0, 90, 100), 100, 100)
        )
        self.assertIsNone(
            allowed_image_box("vehicle.truck", (0, 0, 112, 100), 100, 100)
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


if __name__ == "__main__":
    unittest.main()
