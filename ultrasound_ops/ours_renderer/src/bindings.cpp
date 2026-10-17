#include <torch/extension.h>

#include "shader/ray_attenuation.cuh"
#include "shader/wrapper_shader.cuh"

torch::Tensor CovarianceDiagnosticCUDA(torch::Tensor scales, torch::Tensor quats);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("CovarianceDiagnosticCUDA", &CovarianceDiagnosticCUDA, "Diagnostic of production world covariance helper");
    m.def("ProjectedSliceForwardCUDA", &ProjectedSliceForwardCUDA, "Ours slice forward (CUDA, mm world)");
    m.def(
        "ProjectedSliceForwardRayTileCUDA",
        &ProjectedSliceForwardRayTileCUDA,
        "Ours slice forward with shared local tile and global ray attenuation branches (CUDA)");
    m.def(
        "ProjectedSliceForwardRayTileProfileCUDA",
        &ProjectedSliceForwardRayTileProfileCUDA,
        "Profile Ours local tile forward and ray attenuation branch timings (CUDA)");
    m.def(
        "ProjectedSliceContributionStatsCUDA",
        &ProjectedSliceContributionStatsCUDA,
        "Ours per-Gaussian contribution and residual responsibility stats (CUDA)");
    m.def("ProjectedSliceBackwardCUDA", &ProjectedSliceBackwardCUDA, "Ours slice backward (CUDA, mm world)");
    m.def("RayTileAttenuationCUDA", &RayTileAttenuationCUDA, "Ours local-global ray attenuation modifier (CUDA)");
    m.def(
        "RayTileAttenuationBackwardFromTileCsrCUDA",
        &RayTileAttenuationBackwardFromTileCsrCUDA,
        "Ours local-global ray attenuation gradient for separated attenuation attributes (CUDA)");
    m.def("RayCsrAttenuationCUDA", &RayCsrAttenuationCUDA, "Ours ray-CSR attenuation modifier (CUDA)");
    m.def(
        "RayCsrAttenuationForwardCUDA",
        &RayCsrAttenuationForwardCUDA,
        "Ours ray-CSR attenuation modifier with reusable CSR mapping (CUDA)");
    m.def(
        "RayCsrPathContributionStatsCUDA",
        &RayCsrPathContributionStatsCUDA,
        "Exact per-Gaussian global-path contribution stats from ray-CSR transmission factors (CUDA)");
    m.def(
        "RayCsrPathContributionStatsFromCsrCUDA",
        &RayCsrPathContributionStatsFromCsrCUDA,
        "Exact per-Gaussian global-path contribution stats from a forward ray CSR (CUDA)");
    m.def(
        "RayCsrAttenuationForwardProfileCUDA",
        &RayCsrAttenuationForwardProfileCUDA,
        "Profile Ours ray-CSR attenuation internals (CUDA)");
    m.def(
        "RayCsrAttenuationBackwardCUDA",
        &RayCsrAttenuationBackwardCUDA,
        "Ours ray-CSR attenuation gradient for separated attenuation attributes (CUDA)");
}
