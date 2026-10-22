// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package smpolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

const g0 = "GPU-ca40ff2c-28b4-e6eb-d2e0-c534167bdb32"
const g1 = "GPU-f2ca882c-91aa-b08f-e4c6-c26f3e69bea4"

func testPolicy() *Policy {
	return &Policy{Version: 1, Rules: []Rule{{Namespace: "models", Container: "main", MatchLabels: map[string]string{"worker": "a"},
		Partitions: []Partition{{Device: g0, Name: "a0", Chunks: 18}, {Device: g1, Name: "a1", Chunks: 36}}}}}
}

func TestResolveRequiresExactAssignmentAndSelector(t *testing.T) {
	p := testPolicy()
	for _, tc := range []struct {
		name    string
		labels  map[string]string
		devices []string
		wantErr bool
	}{
		{"reordered CDI", map[string]string{"worker": "a"}, []string{"k8s.device-plugin.nvidia.com/gpu=" + g1, g0}, false},
		{"missing GPU", map[string]string{"worker": "a"}, []string{g0}, true},
		{"duplicate GPU", map[string]string{"worker": "a"}, []string{g0, g0}, true},
		{"numeric index", map[string]string{"worker": "a"}, []string{"0", "1"}, true},
		{"missing labels", nil, []string{g0, g1}, true},
		{"wrong worker", map[string]string{"worker": "b"}, []string{g0, g1}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parts, err := p.Resolve("models", "main", tc.labels, tc.devices)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Resolve = %v, %v", parts, err)
			}
			if !tc.wantErr && (len(parts) != 2 || parts[0].Device != g0 || parts[1].Device != g1) {
				t.Fatal(parts)
			}
		})
	}
	if parts, err := p.Resolve("unrelated", "main", nil, nil); err != nil || len(parts) != 0 {
		t.Fatal(parts, err)
	}
	p.Rules = append(p.Rules, p.Rules[0])
	if _, err := p.Resolve("models", "main", map[string]string{"worker": "a"}, []string{g0, g1}); err == nil {
		t.Fatal("ambiguous policy accepted")
	}
}

func TestPolicyParsingRejectsUnsafeInput(t *testing.T) {
	b, _ := json.Marshal(testPolicy())
	for _, s := range []string{
		string(b) + " {}", "null", `{"version":1,"rules":[],"typo":true}`,
		strings.Replace(string(b), `"chunks":18`, `"chunks":0`, 1),
		strings.Replace(string(b), g0, "0", 1),
		strings.Replace(string(b), `"name":"a0"`, `"name":"none"`, 1),
	} {
		if _, err := Parse(s); err == nil {
			t.Fatalf("accepted %s", s)
		}
	}
	if p, err := Parse(string(b)); err != nil || p == nil {
		t.Fatal(p, err)
	}
	if p, err := Parse(""); err != nil || p != nil {
		t.Fatal(p, err)
	}
}
