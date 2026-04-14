# H1_2 Asset Staging

This directory is the Isaac-side landing zone for the `h1_2_handless` asset bundle.

Current source files live in `legged_lab/assets/h1_2_description/`:

- `h1_2_handless.urdf`
- `h1_2_handless.xml`
- `meshes/*.STL`

Generated artifacts:

- `h1_2.usd`
- `.asset_hash`

Important notes:

- `config.yaml` points the USD conversion pipeline at `h1_2_handless.urdf`.
- The URDF keeps using the source-relative `meshes/` directory under `legged_lab/assets/h1_2_description/`.
- If the source URDF or meshes change, rerun the URDF-to-USD conversion to refresh `h1_2.usd`.
