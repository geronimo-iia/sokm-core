# Decision: Both modalities share SokmConfig + KernelConfig

## Problem

`GestaltKernelGraph` (K³) has two modalities. The question is whether each
modality should have independent configuration or share the same config.

## Decision

Both K¹ and K² use the same `SokmConfig` + `KernelConfig`. No per-modality config.

## Why

Hoya treats the two modalities symmetrically (pp. 60–79). Introducing per-modality
config parameters without Hoya's backing would be an undocumented extension that
future implementers would need to reason about.

Symmetric config also simplifies the API: `GestaltConfig` wraps one `SokmConfig`
+ one `KernelConfig` + one `CrossSokmConfig` rather than doubling the config tree.

## What was rejected

Per-modality SokmConfig/KernelConfig: more complex API, no Hoya backing,
no clear use case in the current scope.
