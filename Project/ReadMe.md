# Preprocessing and data products

The preprocessing code in `Preprocessing/` implements sensor calibration,
projection, rasterization, and train-split normalization. For complete
terminal-based setup and execution, see [EXECUTION.md](EXECUTION.md).

This code is intended solely for research and study purposes and is not for
commercial use.

## Processing stages

- **Calibration and projection:** LiDAR and radar points are transformed
  through sensor, vehicle, and camera frames using nuScenes extrinsics,
  intrinsics, and ego poses.
- **Image canvas:** each 1600x900 camera image is resized to 640x360 and
  vertically padded to a 640x640 canvas with value 114. Sensor image maps use
  the same canvas.
- **LiDAR BEV:** one height channel is generated at 640x640 for `front`
  (X=0..40 m, Y=-20..20 m, 0.0625 m cells) and `full360` (X,Y=-40..40 m,
  0.125 m cells) views.
- **Radar:** filtered radar points are projected onto camera-aligned maps;
  no radar BEV map is generated.
- **Normalization:** statistics are computed from the training split and
  stored in `norm_stats.json`. Empty LiDAR/radar pixels remain zero.
- **Images and labels:** images are stored as uint8 JPEG files; YOLO labels
  use `class cx cy width height` with coordinates normalized to the common
  640x640 canvas.

## Outputs

Preprocessing writes to `Project/Output/`:

| Path | Contents |
| --- | --- |
| `images/` | `{sample_token}_{camera}.jpg` |
| `labels/` | `{sample_token}_{camera}.txt` |
| `sensors/` | `{sample_token}.npz` with sensor maps, BEV maps, points, calibration, and object features |
| `norm_stats.json` | Training-split normalization statistics |
| `splits.json` | Manifest and scene-grouped sample split mappings |
| `tracks.json` | Instance tracks used by the temporal classifier |

The five camera channels are `CAM_FRONT`, `CAM_FRONT_LEFT`,
`CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, and `CAM_BACK_RIGHT`.

| Class ID | Class |
| ---: | --- |
| 0 | bicycle |
| 1 | bus |
| 2 | car |
| 3 | motorcycle |
| 4 | person |
| 5 | truck |

The scene-grouped split is the split used for model training and evaluation.
Samples from a scene should not cross train, validation, and test partitions.

For code clarification or to report a technical issue, contact the author at
[rmallava@gitam.in](mailto:rmallava@gitam.in).
