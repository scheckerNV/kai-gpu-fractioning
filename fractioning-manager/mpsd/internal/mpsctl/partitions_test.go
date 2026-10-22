// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsctl

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"
)

const partitionGPU = "GPU-ca40ff2c-28b4-e6eb-d2e0-c534167bdb32"
const partitionHeader = "SERVER,DEVICE,NAME,CHUNKS,SM COUNT,INCLUDE REMAINDER,SHAREABLE,MAX CHUNKS,CONTRACT TIMEOUT,STATUS,VISIBLE UUID\n"

func TestEnsurePartitionVerifiesExistingAndDoesNotResize(t *testing.T) {
	for _, tc := range []struct {
		name, row string
		bad       bool
	}{
		{"matching", fmt.Sprintf("shared,%s,a0,18,36,false,false,-,-,created,token\n", partitionGPU), false},
		{"wrong size", fmt.Sprintf("shared,%s,a0,36,72,false,false,-,-,created,token\n", partitionGPU), true},
		{"shareable", fmt.Sprintf("shared,%s,a0,18,36,false,true,-,-,created,token\n", partitionGPU), true},
		{"malformed", "shared,missing-columns\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &recorder{out: []byte(partitionHeader + tc.row)}
			c := New(Options{Binary: "mps", ControlPort: "3", Run: r.run})
			err := c.EnsureSMPartition(context.Background(), "shared", partitionGPU, "a0", 18)
			if (err != nil) != tc.bad {
				t.Fatal(err)
			}
			if len(r.calls) != 1 {
				t.Fatal("existing partition was mutated", r.calls)
			}
		})
	}
}

func TestCreateMissingPartitionAndReadBack(t *testing.T) {
	created := false
	calls := 0
	c := New(Options{Binary: "mps", ControlPort: "3", Run: func(_ context.Context, _ string, args ...string) ([]byte, error) {
		calls++
		if slices.Contains(args, "create") {
			want := []string{"-p", "3", "sm-partition", "create", "a0", "--server=shared", "--device=" + partitionGPU, "--chunks=18"}
			if !slices.Equal(args, want) {
				t.Fatal(args)
			}
			created = true
			return nil, nil
		}
		out := partitionHeader
		if created {
			out += fmt.Sprintf("shared,%s,a0,18,36,false,false,-,-,created,token\n", partitionGPU)
		}
		return []byte(out), nil
	}})
	if err := c.EnsureSMPartition(context.Background(), "shared", partitionGPU, "a0", 18); err != nil {
		t.Fatal(err)
	}
	if calls != 3 || !created {
		t.Fatal(calls, created)
	}
}

func TestNamespaceDeviceBindingUsesCanonicalDeviceAndRejectsDuplicateReadback(t *testing.T) {
	r := &recorder{}
	c := New(Options{Binary: "mps", ControlPort: "3", Run: r.run})
	if err := c.SetNamespaceSMPartition(context.Background(), "shared", "kai_a", partitionGPU, "a0"); err != nil {
		t.Fatal(err)
	}
	want := []string{"-p", "3", "namespace", "set", "kai_a", "--server=shared", "--sm-partition=a0", "--device=" + partitionGPU}
	if !slices.Equal(r.call(t, 0), want) {
		t.Fatal(r.call(t, 0))
	}
	r.out = []byte(strings.Repeat(partitionGPU+":a0\n", 2))
	if _, err := c.NamespaceSMPartitions(context.Background(), "shared", "kai_a"); err == nil {
		t.Fatal("duplicate GPU binding accepted")
	}
}
