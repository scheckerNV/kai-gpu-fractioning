// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package smpolicy describes administrator-assigned static MPS partitions.
// It does not allocate scheduler resources or infer physical GPU indices.
package smpolicy

import (
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strings"
)

// Partition is a named, non-shareable partition on a canonical physical GPU.
// Chunks use the device's MPS granularity, not a portable SM percentage.
type Partition struct {
	Device string `json:"device"`
	Name   string `json:"name"`
	Chunks int    `json:"chunks"`
}

// Rule selects a container independently of its changing pod/container ID.
type Rule struct {
	Namespace   string            `json:"namespace"`
	Container   string            `json:"container"`
	MatchLabels map[string]string `json:"matchLabels"`
	Partitions  []Partition       `json:"partitions"`
}

// Policy is loaded from an administrator-owned ConfigMap at mpsd startup.
// A namespace/container covered by rules must match exactly one label selector.
type Policy struct {
	Version int    `json:"version"`
	Rules   []Rule `json:"rules"`
}

var gpuUUID = regexp.MustCompile(`^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)
var partitionName = regexp.MustCompile(`^[a-z0-9_]+$`)

// Validate rejects unsupported selectors and inconsistent partition definitions.
func (p *Policy) Validate() error {
	if p == nil {
		return nil
	}
	if p.Version != 1 || len(p.Rules) == 0 {
		return fmt.Errorf("SM partition policy requires version=1 and nonempty rules")
	}
	definitions := map[string]int{}
	for i, r := range p.Rules {
		if r.Namespace == "" || r.Container == "" || len(r.MatchLabels) == 0 || len(r.Partitions) == 0 {
			return fmt.Errorf("SM partition rule %d requires namespace, container, matchLabels and partitions", i)
		}
		for k, v := range r.MatchLabels {
			if k == "" || v == "" {
				return fmt.Errorf("SM partition rule %d has an empty label", i)
			}
		}
		devices := map[string]bool{}
		for _, part := range r.Partitions {
			if !gpuUUID.MatchString(part.Device) || !partitionName.MatchString(part.Name) || part.Name == "none" || part.Chunks < 1 {
				return fmt.Errorf("invalid SM partition in rule %d: %+v", i, part)
			}
			if devices[part.Device] {
				return fmt.Errorf("rule %d repeats GPU %s", i, part.Device)
			}
			devices[part.Device] = true
			key := part.Device + "/" + part.Name
			if chunks, exists := definitions[key]; exists && chunks != part.Chunks {
				return fmt.Errorf("inconsistent chunks for %s", key)
			}
			definitions[key] = part.Chunks
		}
	}
	return nil
}

// Parse accepts empty input as disabled. Unknown fields and trailing JSON fail.
func Parse(input string) (*Policy, error) {
	if strings.TrimSpace(input) == "" {
		return nil, nil
	}
	var p Policy
	d := json.NewDecoder(strings.NewReader(input))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil {
		return nil, fmt.Errorf("SM partition policy: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("SM partition policy must contain exactly one JSON object")
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &p, nil
}

// CanonicalDevice accepts the canonical UUID or the CDI assignment KAI emits.
func CanonicalDevice(device string) (string, error) {
	device = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(device), "k8s.device-plugin.nvidia.com/gpu="))
	if !gpuUUID.MatchString(device) {
		return "", fmt.Errorf("static SM partition requires a canonical physical GPU UUID, got %q", device)
	}
	return device, nil
}

// Resolve requires an exact device set. Extra/missing devices cannot fall back
// to a dynamic namespace on a partly configured TP worker.
func (p *Policy) Resolve(namespace, container string, labels map[string]string, assigned []string) ([]Partition, error) {
	if p == nil {
		return nil, nil
	}
	covered := false
	var matched []Rule
	for _, r := range p.Rules {
		if r.Namespace != namespace || r.Container != container {
			continue
		}
		covered = true
		match := true
		for k, v := range r.MatchLabels {
			if labels[k] != v {
				match = false
				break
			}
		}
		if match {
			matched = append(matched, r)
		}
	}
	if !covered {
		return nil, nil
	}
	if len(matched) != 1 {
		return nil, fmt.Errorf("SM partition policy for %s/%s matched %d rules, expected exactly one", namespace, container, len(matched))
	}
	devices := map[string]bool{}
	for _, raw := range assigned {
		uuid, err := CanonicalDevice(raw)
		if err != nil {
			return nil, err
		}
		if devices[uuid] {
			return nil, fmt.Errorf("duplicate assigned GPU %s", uuid)
		}
		devices[uuid] = true
	}
	parts := slices.Clone(matched[0].Partitions)
	if len(devices) != len(parts) {
		return nil, fmt.Errorf("SM partition policy device count %d differs from assigned count %d", len(parts), len(devices))
	}
	for _, part := range parts {
		if !devices[part.Device] {
			return nil, fmt.Errorf("SM partition policy GPU %s was not assigned to the container", part.Device)
		}
	}
	slices.SortFunc(parts, func(a, b Partition) int { return strings.Compare(a.Device, b.Device) })
	return parts, nil
}

// Definitions returns all partitions in stable order, once per device/name.
func (p *Policy) Definitions() []Partition {
	if p == nil {
		return nil
	}
	seen := map[string]bool{}
	var parts []Partition
	for _, r := range p.Rules {
		for _, part := range r.Partitions {
			key := part.Device + "/" + part.Name
			if !seen[key] {
				parts = append(parts, part)
				seen[key] = true
			}
		}
	}
	slices.SortFunc(parts, func(a, b Partition) int { return strings.Compare(a.Device+"/"+a.Name, b.Device+"/"+b.Name) })
	return parts
}
