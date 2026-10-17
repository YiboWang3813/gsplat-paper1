# gaussianplie_renderer

GaussianPile renderer package. The package exposes GaussianPile-specific config, reference math, and a
CUDA-backed renderer API used by `methods/gaussianpile`.

The initial CUDA-backed implementation delegates tile/additive GPU rasterization to the existing
`ours_renderer` extension while keeping the GaussianPile-facing API and reference math local to this package.

