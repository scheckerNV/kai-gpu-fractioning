// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsns

import (
	"context"
	"fmt"
	"maps"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/smpolicy"
)

// This extension is optional: legacy scalar-only Control implementations keep
// working when no static policy or saved static lease is present.
type partitionControl interface {
	EnsureSMPartition(context.Context, string, string, string, int) error
	SetNamespaceSMPartition(context.Context, string, string, string, string) error
	NamespaceSMPartitions(context.Context, string, string) (map[string]string, error)
}

func (m *Manager) ensurePartitions(ctx context.Context, parts []smpolicy.Partition) error {
	if len(parts) == 0 {
		return nil
	}
	c, ok := m.control.(partitionControl)
	if !ok {
		return fmt.Errorf("per-device namespace control is unavailable")
	}
	for _, p := range parts {
		if err := c.EnsureSMPartition(ctx, m.server, p.Device, p.Name, p.Chunks); err != nil {
			return fmt.Errorf("ensuring SM partition %s/%s: %w", p.Device, p.Name, err)
		}
	}
	return nil
}

// No two live/pending leases may claim the same static partition. A new pod
// waits for the previous container's normal release; rolling replacement must
// not silently run two model workers in one assigned share.
func (m *Manager) checkPartitionClaims(containerID string, parts []smpolicy.Partition) error {
	for id, lease := range m.leases {
		if id == containerID {
			continue
		}
		for _, wanted := range parts {
			for _, held := range lease.SMPartitions {
				if wanted.Device == held.Device && wanted.Name == held.Name {
					return fmt.Errorf("SM partition %s/%s is still leased to container %s", wanted.Device, wanted.Name, id)
				}
			}
		}
	}
	return nil
}

// apply is false on a retried create: active clients prohibit namespace set.
// Exact readback is required even on retries; saved state alone is not proof.
func (m *Manager) bindPartitions(ctx context.Context, namespace string, parts []smpolicy.Partition, apply bool) error {
	if len(parts) == 0 {
		return nil
	}
	c, ok := m.control.(partitionControl)
	if !ok {
		return fmt.Errorf("per-device namespace control is unavailable")
	}
	expected := map[string]string{}
	for _, p := range parts {
		expected[p.Device] = p.Name
		if apply {
			if err := c.SetNamespaceSMPartition(ctx, m.server, namespace, p.Device, p.Name); err != nil {
				return fmt.Errorf("assigning namespace %s GPU %s: %w", namespace, p.Device, err)
			}
		}
	}
	actual, err := c.NamespaceSMPartitions(ctx, m.server, namespace)
	if err != nil {
		return err
	}
	if !maps.Equal(actual, expected) {
		return fmt.Errorf("namespace %s SM partition readback %v differs from %v", namespace, actual, expected)
	}
	return nil
}
