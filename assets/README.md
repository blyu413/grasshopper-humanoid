# Robot assets

[Project overview](../README.md)

## Current training model

`robots/grasshopper_torso/` contains the Grasshopper lower-body/torso model used by
the standing and walking tasks:

- `grasshopper_lower_body.urdf`: source of joint limits and capsule-foot geometry.
- `meshes/`: the URDF's visual and collision mesh resources.
- `usd/grasshopper_lower_body/grasshopper_lower_body.usda`: base mesh-foot USD.
- `usd/grasshopper_lower_body/payloads/`: the complete base USD dependency set,
  including geometry, robot/material layers and MuJoCo/PhysX physics variants.
- `grasshopper_lower_body_capsule_feet.usda`: the default capsule-foot override,
  referencing the versioned base USD with a relative path.
- `grasshopper_asset_cfg.py`: compatibility import for the configuration now under `src/nest/`.

The nine base USD files were copied byte-for-byte from the existing
`outputs/grasshopper_torso/free_base_usd/grasshopper_lower_body/` export. The original
generated copy remains local for historical experiments. Current source and the
capsule override reference the versioned copy in `assets/`.

To rebuild the capsule override after changing the URDF foot geometry:

```bash
.venv/bin/python scripts/build_grasshopper_capsule_feet.py
.venv/bin/python -m unittest discover -s tests -p 'test_grasshopper_foot_collisions.py' -v
```

This rebuilds the override, not the base robot conversion. Preserve the complete
base USD directory when updating or distributing it. Its metadata identifies
`URDF USD Converter v0.3.2`; the original base-conversion command is not recorded
in the current source scripts.

`terrains/flat_plane.usda` is the reusable flat terrain asset.

## Imported model descriptions

`robots/grasshopper_torso_arm/` contains separate imported descriptions:

| Directory | Contents |
| --- | --- |
| `grasshopper_latest_tesollo_description/` | Full-body Grasshopper with Tesollo hands and bundled meshes |
| `pedestal_latest_tesollo_description/` | Pedestal variant with bundled meshes |
| `grasshopper_latest_tesollo_desription_mjcf/` | Converted MJCF world, conversion intermediates, debug snapshots and mesh assets |

The imported directory spelling is preserved for compatibility. These models do
not replace the lower-body training asset. `scripts/compare_grasshopper_models.py`
compares the training URDF against the imported full-body Grasshopper description.

Machine-specific `/home/...`, temporary export paths and ROS package mesh references
in the imported URDF/XML files have been replaced with relative references to the
bundled resources. The MJCF world resolves its meshes from this directory. Geometry,
mass, inertia, joint limits and control parameters are unchanged by the path cleanup.
Conversion/debug files are retained as reference material; resolving their files
does not establish physical equivalence with the training model.

The aggregate URDF's ROS control `mujoco_model` parameter now names `world.xml`.
When using that ROS integration, run from the MJCF directory or have the launch
configuration resolve this parameter to the world file. ROS control itself was
not exercised during this file cleanup.

## Source and license notes

The imported assets contain NEST robot geometry and meshes under names including
`tesollo_model` and `realsense2_description`. No upstream revision or license files
were supplied with these asset bundles. The source archive name is
`nest_description_for_nov`; an upstream repository URL has not been established.

Keep the asset provenance and license question open until the owners provide the
original sources and terms. This cleanup does not assign a new license to imported
geometry or establish permission for public redistribution.
