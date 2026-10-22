// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsns

import (
	"context"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"testing"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/smpolicy"
)

const testGPU0 = "GPU-ca40ff2c-28b4-e6eb-d2e0-c534167bdb32"
const testGPU1 = "GPU-f2ca882c-91aa-b08f-e4c6-c26f3e69bea4"

type staticControl struct {
	*fakeControl
	parts         map[string]int
	bindings      map[string]map[string]string
	failDevice    string
	readbackWrong bool
	setCount      int
}

func newStaticControl() *staticControl {
	return &staticControl{fakeControl: newFakeControl(), parts: map[string]int{}, bindings: map[string]map[string]string{}}
}
func (f *staticControl) EnsureSMPartition(_ context.Context, server, device, name string, chunks int) error {
	key := server + "/" + device + "/" + name
	if old, ok := f.parts[key]; ok && old != chunks {
		return errors.New("partition size conflict")
	}
	f.parts[key] = chunks
	return nil
}
func (f *staticControl) SetNamespaceSMPartition(_ context.Context, server, name, device, partition string) error {
	if device == f.failDevice {
		return errors.New("driver rejected assignment")
	}
	key := f.key(server, name)
	if f.busy[key] {
		return errors.New("active client")
	}
	if f.bindings[key] == nil {
		f.bindings[key] = map[string]string{}
	}
	f.bindings[key][device] = partition
	f.setCount++
	return nil
}
func (f *staticControl) NamespaceSMPartitions(_ context.Context, server, name string) (map[string]string, error) {
	if f.readbackWrong {
		return map[string]string{testGPU0: "wrong"}, nil
	}
	return maps.Clone(f.bindings[f.key(server, name)]), nil
}
func staticPolicy() *smpolicy.Policy {
	return &smpolicy.Policy{Version: 1, Rules: []smpolicy.Rule{{Namespace: "models", Container: "main", MatchLabels: map[string]string{"worker": "a"}, Partitions: []smpolicy.Partition{{Device: testGPU0, Name: "a0", Chunks: 18}, {Device: testGPU1, Name: "a1", Chunks: 36}}}}}
}
func staticRequest(id string) Request {
	return Request{ContainerID: id, ActiveThreadPercent: 50, GPUUUIDs: []string{testGPU1, "k8s.device-plugin.nvidia.com/gpu=" + testGPU0}, Namespace: "models", Container: "main", PodLabels: map[string]string{"worker": "a"}}
}
func staticManager(t *testing.T, f *staticControl, state string) *Manager {
	t.Helper()
	m, err := New(Options{Control: f, Server: "shared", StatePath: state, SMPartitionPolicy: staticPolicy(), Live: func(string) (bool, error) { return true, nil }})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestStaticLifecyclePersistsRestoresAndRecreates(t *testing.T) {
	ctx := context.Background()
	f := newStaticControl()
	path := filepath.Join(t.TempDir(), "state.json")
	m := staticManager(t, f, path)
	a, err := m.Provision(ctx, staticRequest("container-a"))
	if err != nil {
		t.Fatal(err)
	}
	if a.ActiveThreadPercent != 100 || len(a.SMPartitions) != 2 || f.setCount != 2 {
		t.Fatal(a, f.setCount)
	}
	f.busy[f.key("shared", a.Namespace)] = true
	if _, err = m.Provision(ctx, staticRequest("container-a")); err != nil {
		t.Fatal("retry should read back without mutating", err)
	}
	if f.setCount != 2 {
		t.Fatal("retry mutated an active namespace")
	}
	if _, err = m.Provision(ctx, staticRequest("duplicate-a")); err == nil {
		t.Fatal("same partition was leased twice")
	}
	// Both daemon and manager lose memory: restoration must use the saved map.
	g := newStaticControl()
	restored := staticManager(t, g, path)
	if err = restored.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if g.bindings[g.key("shared", a.Namespace)][testGPU1] != "a1" || len(g.parts) != 2 {
		t.Fatal(g.bindings, g.parts)
	}
	if deleted, err := restored.Release(ctx, "container-a"); err != nil || !deleted {
		t.Fatal(deleted, err)
	}
	b, err := restored.Provision(ctx, staticRequest("recreated-a"))
	if err != nil {
		t.Fatal(err)
	}
	if b.Namespace == a.Namespace || len(b.SMPartitions) != 2 {
		t.Fatal(a, b)
	}
}

func TestStaticAssignmentReadbackAndPersistenceFailuresBlockStartup(t *testing.T) {
	for _, tc := range []string{"second GPU set fails", "wrong readback", "state not writable"} {
		t.Run(tc, func(t *testing.T) {
			f := newStaticControl()
			path := filepath.Join(t.TempDir(), "state.json")
			if tc == "second GPU set fails" {
				f.failDevice = testGPU1
			}
			if tc == "wrong readback" {
				f.readbackWrong = true
			}
			if tc == "state not writable" {
				dir := filepath.Join(t.TempDir(), "file")
				if err := os.WriteFile(dir, []byte("x"), 0600); err != nil {
					t.Fatal(err)
				}
				path = filepath.Join(dir, "state.json")
			}
			m := staticManager(t, f, path)
			if _, err := m.Provision(context.Background(), staticRequest("container-a")); err == nil {
				t.Fatal("provision succeeded with incomplete enforcement/persistence")
			}
		})
	}
}

func TestStaticRetryRejectsChangedDaemonMap(t *testing.T) {
	f := newStaticControl()
	m := staticManager(t, f, filepath.Join(t.TempDir(), "state.json"))
	ctx := context.Background()
	if _, err := m.Provision(ctx, staticRequest("container-a")); err != nil {
		t.Fatal(err)
	}
	f.readbackWrong = true
	if _, err := m.Provision(ctx, staticRequest("container-a")); err == nil {
		t.Fatal("trusted stale lease without live readback")
	}
}

func TestStaticCorruptOwnershipStateFailsEveryRetry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(path, []byte("broken"), 0600); err != nil {
		t.Fatal(err)
	}
	f := newStaticControl()
	m := staticManager(t, f, path)
	for i := 0; i < 2; i++ {
		if _, err := m.Provision(context.Background(), staticRequest("container-a")); err == nil {
			t.Fatal("corrupt ownership state accepted")
		}
	}
	if f.setCount != 0 || len(f.parts) != 0 {
		t.Fatal("mutated daemon with unknown partition ownership")
	}
}
