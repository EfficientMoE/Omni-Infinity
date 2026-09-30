# C5 denoise-step cache

The C5 cache is explicitly opt-in and requires coefficients calibrated for
the target model. No published TeaCache or cache-dit table covers MiniMax-H3.
This plan does not add a CLIP, SSIM, or PSNR gate.

The cache is local to one generation and wraps only the transformer's forward
method. Never publish H3 text K/V from a denoise step.
mode="output" is the H3 default. Residual mode is available only when output
and named-input tensor shapes match.
