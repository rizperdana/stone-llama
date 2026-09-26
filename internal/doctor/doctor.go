// Package doctor probes the machine for stone-llama's hard prerequisites:
// an NVIDIA GPU, a usable driver, and which runtime extra (cu12/cu13)
// applies. It only reads state — it never changes anything.
package doctor

import (
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

type GPU struct {
	Name    string
	VRAMMiB int
	Driver  string
}

type Report struct {
	GPUs []GPU
	// Extra is the TabbyAPI install extra this machine should use:
	// "cu13" (driver >= 580), "cu12" (driver >= 570), or "" if unusable.
	Extra string
	// Note explains a refusal (Extra == "") or warns (e.g. cu12 chosen).
	Note string
}

func (r Report) Ready() bool { return r.Extra != "" }

// Probe queries nvidia-smi. Missing/broken nvidia-smi is not an error —
// it is a negative verdict.
func Probe() (Report, error) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return Report{Note: "no nvidia-smi executable"}, nil
	}
	out, err := exec.Command(path, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return Report{Note: fmt.Sprintf("nvidia-smi failed: %v: %s", err, strings.TrimSpace(string(out)))}, nil
	}

	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if g, ok := parseLine(line); ok {
			gpus = append(gpus, g)
		}
	}
	if len(gpus) == 0 {
		return Report{Note: "nvidia-smi returned no GPUs"}, nil
	}

	extra, note := extraForDriver(gpus[0].Driver)
	return Report{GPUs: gpus, Extra: extra, Note: note}, nil
}

// parseLine parses one `name, memory.total, driver_version` CSV row.
// GPU names may contain commas, so fields are split from the right.
func parseLine(line string) (GPU, bool) {
	parts := strings.Split(line, ", ")
	if len(parts) < 3 {
		return GPU{}, false
	}
	driver := strings.TrimSpace(parts[len(parts)-1])
	mem := strings.TrimSpace(parts[len(parts)-2])
	name := strings.TrimSpace(strings.Join(parts[:len(parts)-2], ", "))
	n, err := strconv.Atoi(mem)
	if err != nil {
		return GPU{}, false
	}
	return GPU{Name: name, VRAMMiB: n, Driver: driver}, true
}

func extraForDriver(ver string) (string, string) {
	major, err := strconv.Atoi(strings.Split(ver, ".")[0])
	if err != nil {
		return "", fmt.Sprintf("unparseable driver version %q", ver)
	}
	switch {
	case major >= 580:
		return "cu13", ""
	case major >= 570:
		return "cu12", fmt.Sprintf("driver %s → runtime extra cu12 (cu13 needs ≥ 580)", ver)
	default:
		return "", fmt.Sprintf("NVIDIA driver %s is too old: stone-llama needs ≥ 570 (cu12) or ≥ 580 (cu13)", ver)
	}
}

// Format renders the report for terminal output.
func (r Report) Format() string {
	var b strings.Builder
	if len(r.GPUs) == 0 {
		b.WriteString("✗ stone-llama needs an NVIDIA GPU (CUDA). None detected on this machine.\n")
		fmt.Fprintf(&b, "  Detected: %s.\n", r.Note)
		b.WriteString("  stone-llama runs ExLlamaV3, which is CUDA-only — no CPU, AMD, or Apple support.\n")
		b.WriteString("  For CPU-only or AMD machines, use ollama with GGUF models instead.\n")
		return b.String()
	}

	for i, g := range r.GPUs {
		if i == 0 {
			fmt.Fprintf(&b, "GPU        %s\n", g.Name)
			fmt.Fprintf(&b, "VRAM       %d MiB\n", g.VRAMMiB)
			fmt.Fprintf(&b, "Driver     %s\n", g.Driver)
		} else {
			fmt.Fprintf(&b, "GPU#%d      %s (%d MiB)\n", i+1, g.Name, g.VRAMMiB)
		}
	}
	if r.Ready() {
		fmt.Fprintf(&b, "Runtime    %s extra\n", r.Extra)
		if r.Note != "" {
			fmt.Fprintf(&b, "Note       %s\n", r.Note)
		}
	} else {
		fmt.Fprintf(&b, "✗ %s\n", r.Note)
	}
	return b.String()
}

// Contender is one process nvidia-smi reports holding GPU compute memory.
type Contender struct {
	PID  int
	Name string
	MiB  int64
}

// Contenders lists processes currently using GPU memory — evidence for a
// confusingly failed spawn. Unlike Probe, a missing nvidia-smi is an
// error here: the caller is explaining a failure, not judging readiness.
func Contenders() ([]Contender, error) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, errors.New("nvidia-smi not found — is the NVIDIA driver installed? run 'stone-llama doctor'")
	}
	out, err := exec.Command(path,
		"--query-compute-apps=pid,process_name,used_gpu_memory",
		"--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi --query-compute-apps failed: %v: %s", err, strings.TrimSpace(string(out)))
	}
	var cs []Contender
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if c, ok := parseContender(line); ok {
			cs = append(cs, c)
		}
	}
	return cs, nil
}

// parseContender parses one `pid, process_name, used_gpu_memory` CSV row;
// process_name may contain commas, so fields split at the edges. Unusable
// rows (e.g. "N/A" memory) are skipped like Probe skips garbage lines.
func parseContender(line string) (Contender, bool) {
	parts := strings.Split(line, ",")
	if len(parts) < 3 {
		return Contender{}, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return Contender{}, false
	}
	mib, err := strconv.ParseInt(strings.TrimSpace(parts[len(parts)-1]), 10, 64)
	if err != nil {
		return Contender{}, false
	}
	name := strings.TrimSpace(strings.Join(parts[1:len(parts)-1], ","))
	return Contender{PID: pid, Name: name, MiB: mib}, true
}

// ContentionError names up to three processes holding the GPU when a
// spawn failed; nil when nothing holds it.
func ContentionError(cs []Contender) error {
	if len(cs) == 0 {
		return nil
	}
	shown := cs
	var more string
	if len(cs) > 3 {
		shown = cs[:3]
		more = fmt.Sprintf(" and %d more", len(cs)-3)
	}
	var b strings.Builder
	for i, c := range shown {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "PID %d (%s) using %d MiB", c.PID, c.Name, c.MiB)
	}
	b.WriteString(more)
	return fmt.Errorf(
		"another process appears to hold the GPU: %s — stop it, wait, or use the free VRAM (evidence: nvidia-smi --query-compute-apps)",
		b.String())
}
