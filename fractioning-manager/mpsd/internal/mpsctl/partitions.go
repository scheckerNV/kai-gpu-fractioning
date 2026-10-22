// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsctl

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// EnsureSMPartition creates a missing partition and verifies its full-device
// identity, size and non-shareable state. Existing partitions are never resized
// or deleted; a conflict blocks provisioning instead of disturbing another user.
func (c *Control) EnsureSMPartition(ctx context.Context, server, device, name string, chunks int) error {
	if err := checkNamespaceArgs(server, name); err != nil {
		return err
	}
	if !strings.HasPrefix(device, "GPU-") || chunks < 1 {
		return fmt.Errorf("invalid SM partition device/chunks")
	}
	check := func() (bool, error) {
		out, err := c.exec(ctx, c.timeout, "sm-partition", "list", "--server="+server, "--all", "--format=csv")
		if err != nil {
			return false, err
		}
		r := csv.NewReader(strings.NewReader(string(out)))
		header, err := r.Read()
		if err != nil {
			return false, fmt.Errorf("SM partition CSV header: %w", err)
		}
		columns := map[string]int{}
		for i, h := range header {
			columns[strings.TrimSpace(h)] = i
		}
		for _, h := range []string{"SERVER", "DEVICE", "NAME", "CHUNKS", "SHAREABLE", "STATUS"} {
			if _, ok := columns[h]; !ok {
				return false, fmt.Errorf("SM partition CSV lacks %s", h)
			}
		}
		found := false
		for {
			row, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				return false, fmt.Errorf("SM partition CSV: %w", err)
			}
			field := func(h string) string { return strings.TrimSpace(row[columns[h]]) }
			if field("SERVER") != server || field("DEVICE") != device || field("NAME") != name {
				continue
			}
			if found {
				return false, fmt.Errorf("duplicate SM partition %s/%s/%s", server, device, name)
			}
			if field("CHUNKS") != strconv.Itoa(chunks) || field("SHAREABLE") != "false" || field("STATUS") != "created" {
				return false, fmt.Errorf("SM partition %s/%s disagrees with requested non-shareable %d chunks: %v", device, name, chunks, row)
			}
			found = true
		}
		return found, nil
	}
	if found, err := check(); err != nil || found {
		return err
	}
	_, createErr := c.exec(ctx, c.timeout, "sm-partition", "create", name, "--server="+server, "--device="+device, "--chunks="+strconv.Itoa(chunks))
	found, err := check()
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("SM partition %s/%s not present after creation (create error: %v)", device, name, createErr)
	}
	return nil
}

// SetNamespaceSMPartition requires the r620 per-device namespace API. Older
// daemons reject the command; there is deliberately no client-env fallback.
func (c *Control) SetNamespaceSMPartition(ctx context.Context, server, namespace, device, partition string) error {
	if err := checkNamespaceArgs(server, namespace); err != nil {
		return err
	}
	if !ValidNamespaceName(partition) || partition == "none" || !strings.HasPrefix(device, "GPU-") {
		return fmt.Errorf("invalid SM partition assignment")
	}
	_, err := c.exec(ctx, c.timeout, "namespace", "set", namespace, "--server="+server, "--sm-partition="+partition, "--device="+device)
	return err
}

var devicePartitionPattern = regexp.MustCompile(`(GPU-[0-9a-fA-F-]+):([a-z0-9_]+)`)

// NamespaceSMPartitions reads back canonical device:name pairs. Empty or
// duplicated/malformed results cannot satisfy the manager's exact-map check.
func (c *Control) NamespaceSMPartitions(ctx context.Context, server, namespace string) (map[string]string, error) {
	if err := checkNamespaceArgs(server, namespace); err != nil {
		return nil, err
	}
	out, err := c.exec(ctx, c.timeout, "namespace", "get", namespace, server, "sm-partitions")
	if err != nil {
		return nil, err
	}
	result := map[string]string{}
	for _, match := range devicePartitionPattern.FindAllStringSubmatch(string(out), -1) {
		if _, exists := result[match[1]]; exists {
			return nil, fmt.Errorf("duplicate namespace SM partition GPU %s", match[1])
		}
		result[match[1]] = match[2]
	}
	return result, nil
}
