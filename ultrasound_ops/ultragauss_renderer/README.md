# ultragauss_renderer

UltraGauss CUDA renderer, architecturally aligned with `ultrags_renderer`.

## Coordinate Policy

- Renderer API inputs are world-space tensors.
- Shader computes Mahalanobis weights in world coordinates directly.
- Mapper may temporarily project into probe coordinates only for candidate filtering.

## Equation Mapping

- Eq. (6)(7): normalized alpha-weighted intensity composition.
- Eq. (8): `alpha_i * exp(-0.5 * delta^T * SigmaInv * delta)`.
- Eq. (13)(14): mapper-side chi-square bounded Gaussian candidate windows.
- Eq. (15): optional shadow approximation via cumulative alpha attenuation.

## Build

```bash
cd ops/ultragauss_renderer
bash rebuild.sh
```
