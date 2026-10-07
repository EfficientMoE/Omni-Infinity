# Security Policy

Omni-Infinity is an early-stage (v0.1 bootstrap) research runtime for
memory-constrained omni-modal inference. There are no versioned releases yet;
security fixes land on `main`.

## Supported versions

| Version | Supported |
|---|---|
| `main` (development) | ✅ |
| Tagged releases | none yet |

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue for a security problem.

- Preferred: open a private report through GitHub's
  [Security Advisories](https://github.com/EfficientMoE/Omni-Infinity/security/advisories/new)
  ("Report a vulnerability").
- Alternatively, reach the EfficientMoE maintainers through the
  [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) organization.

Please include a description, the affected version or commit, reproduction
steps, and the impact. We will acknowledge the report and coordinate a fix and
disclosure timeline with you.

## Operational security note

The job-serving and streaming server (`python -m omni_infinity.serve`) has
**no built-in authentication or authorization**. By default it binds
`127.0.0.1` (`OMNI_HOST`), so it is not reachable off the host. Before exposing
it on a shared or untrusted network:

- keep it bound to `127.0.0.1`, or place it behind an authenticating reverse
  proxy or an SSH tunnel;
- treat generation prompts, uploaded first/last frames, and produced artifacts
  as untrusted input and output under your own access controls.

## Model weights and licenses

Model weights (for example `MiniMaxAI/MiniMax-H3` and `OpenVDN/vdn-minimax-h3`)
are distributed under their own licenses — such as the MiniMax H3 Community
License — separate from this repository's Apache-2.0 **code** license. Review
and accept those terms before downloading or serving the weights.
