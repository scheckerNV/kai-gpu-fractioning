# Per-device namespace defaults (experimental)

This opt-in extension binds named static MPS partitions during the NRI
CreateContainer hook. It requires the r620 namespace per-device default API;
the tested development driver is620.69. Legacy scalar ATP deployments remain
unchanged when no policy is configured. This does not add asymmetric scheduling
to KAI or turn MPS into a general security boundary.

An administrator creates an immutable/versioned ConfigMap in the GPU Fractioning
operator namespace. Its JSON policy contains version1 and rules with namespace,
container, matchLabels and partitions. Each partition names a canonical physical
GPU UUID, an MPS partition name and its positive chunk count. Chunk granularity
is device-specific; on the tested A100,18/36 chunks means36/72SMs. Example:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: static-sm-policy-v1
  namespace: gpu-fractioning
data:
  policy.json: |
    {"version":1,"rules":[{"namespace":"models","container":"main","matchLabels":{"model-worker":"a"},"partitions":[{"device":"GPU-00000000-0000-0000-0000-000000000001","name":"worker_a_gpu0","chunks":18}]}]}
---
apiVersion: gpu-fractioning.kai.scheduler/v1alpha1
kind: GpuFractioningConfig
metadata:
  name: default
spec:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  mpsDaemon:
    smPartitionPolicy:
      name: static-sm-policy-v1
      key: policy.json
```

The operator passes the required ConfigMap key as MPS_SM_PARTITION_POLICY.
Missing ConfigMaps cannot silently disable the policy. Invalid JSON disables
namespace provisioning, so SM-sharing container startup fails. Namespace and
container pairs covered by rules must match exactly one selector. Assigned
GPU UUIDs (including KAI's CDI spelling) must match the policy device set exactly.
Numeric GPU ordinals are refused. The policy is specific to the node's physical
GPUs; this initial implementation is not a multi-node GPU placement API.

mpsd verifies or creates all configured, non-shareable static partitions, sets
namespace ATP100, assigns each device default and reads back the exact map.
Only then does fractiond receive the socket and allow creation to finish.
fractiond injects client ATP100 for static workers, avoiding a scalar reporting
ceiling, while memory injection remains unchanged. The model needs no partition
selection environment variable.

Device maps are stored in the existing host-backed namespace lease file.
A replacement container receives a newly derived namespace and the same stable
label-selected map. Normal drain/release must finish before another container
can claim the same partition. Creation retries read back existing bindings
without attempting forbidden mutations on active clients. Missing driver API,
partial assignments, wrong readback, corrupt ownership state and failed durable
state writes block static worker startup. MPS reconciliation can reconstruct
partitions and namespace mappings from saved leases; transparent recovery of
already running CUDA applications after an MPS failure is not promised.

Policy changes require affected clients to stop, a new ConfigMap reference and
an mpsd rollout. Partitions are never resized or deleted by this extension;
existing conflicting definitions fail. KAI reservations still need to account
for the workload separately. In the two-worker reproducer, each reserves50%
and19GiB perGPU, while static compute maps are36/72 and72/36SMs. The pair totals
108SMs and38GiB perGPU. These fixed profiles support one live container per map.
GPU-memory limits still come from the existing container-toolkit hook; unequal
per-GPU memory automation is not added by this compute-binding extension.

Unit tests cover policy/device validation, failed assignment/readback/persistence,
lease restoration, replacement ownership and NRI/operator wiring. The experiment
report records separately which lifecycle operations were verified on real GPUs.
