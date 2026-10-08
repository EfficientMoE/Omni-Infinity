# C5 denoise-step cache

The C5 cache is explicitly opt-in and requires coefficients calibrated for
the target model. No published TeaCache or cache-dit table covers MiniMax-H3.
This plan does not add a CLIP, SSIM, or PSNR gate.

The cache is local to one generation and wraps only the transformer's forward
method. Never publish H3 text K/V from a denoise step.
mode="output" is the H3 default. Residual mode is available only when output
and named-input tensor shapes match.

C5 v2 can select the raw, TeaCache timestep-modulated, or FBCache first-block
residual indicator. TeaCache normally accumulates polynomial-rescaled deltas
until the threshold triggers a compute; approximation is either reuse or
`taylor1`. FBCache normally uses its per-call residual threshold without an
accumulator. The cache still refuses to run without model-specific calibrated
coefficients. Its Python skip decision remains outside compiled/graphed
regions, and cached steps bypass those regions.
