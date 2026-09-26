// Package serve runs the stone-llama daemon: it supervises the pinned
// TabbyAPI child (or attaches to an existing OpenAI-compatible server),
// reverse-proxies with zero-buffer SSE, and keeps single-daemon state
// (daemon.json + flock) per ARCHITECTURE.md §2 and §7.
//
// The daemon owns the only write side of daemon.json and the child
// lifecycle; clients (ps/run) read state and talk to the /-/ control
// endpoints. Downstream auth is ours (token in daemon.json 0600,
// optional on loopback, required otherwise); upstream auth is injected
// from a file and never appears in argv, logs, or stdout.
package serve

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/rizperdana/stone-llama/internal/autofit"
	"github.com/rizperdana/stone-llama/internal/config"
	"github.com/rizperdana/stone-llama/internal/doctor"
	"github.com/rizperdana/stone-llama/internal/fslock"
)

// Sentinel outcomes the CLI maps to friendly text/exit codes.
var (
	ErrAlreadyServing = errors.New("stone-llama is already serving here")
	ErrPortInUse      = errors.New("the configured port is used by another process")
	ErrStartBusy      = errors.New("another stone-llama process is starting the daemon")
)

// errCleanExit names awaitReady's (died, nil) outcome: the child
// exited with status 0 before it ever became ready.
var errCleanExit = errors.New("exited cleanly (status 0) before readiness")

const (
	healthzPath     = "/healthz"
	healthzBody     = "stone-llama\n"
	markerHeader    = "X-Stone-Llama"
	statusPath      = "/-/status"
	loadPath        = "/-/load"
	unloadPath      = "/-/unload"
	defaultReady    = 120 * time.Second
	defaultRetry    = time.Second
	killGrace       = 5 * time.Second
	childRestarts   = 3
	vramSampleEvery = 5 * time.Second
	maxErrBody      = 64 << 10
)

// Options configures Serve. Zero values get defaults (tests override the
// seams below; the CLI never touches them).
type Options struct {
	Host            string // downstream bind host
	Port            int    // downstream bind port
	Attach          string // upstream base URL; "" = supervise a child
	RuntimeDir      string // setup runtime root (venv + tabbyAPI checkout)
	ModelsDir       string
	DataDir         string         // daemon.json + flock + logs
	UpstreamKeyFile string         // file whose CONTENTS become the upstream Bearer (file-only secret)
	PinnedCommit    string         // for the startup banner ("" = omit)
	Fit             config.Autofit // autofit knobs for /-/load

	Stdout, Stderr io.Writer

	ReadyTimeout time.Duration
	RetryDelay   time.Duration
	Now          func() time.Time

	// test seams
	spawn      func(runtimeDir, cfgPath, logPath string, port int) (*child, error)
	probe      func(ctx context.Context, port int) error
	procCmd    func(pid int) string
	contenders func() ([]doctor.Contender, error)
	probeGPU   func() (doctor.Report, error)
}

func (o *Options) withDefaults() {
	if o.Host == "" {
		o.Host = "127.0.0.1"
	}
	if o.RuntimeDir == "" {
		o.RuntimeDir = filepath.Join(o.DataDir, "runtime")
	}
	if o.Stdout == nil {
		o.Stdout = io.Discard
	}
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if o.ReadyTimeout <= 0 {
		o.ReadyTimeout = defaultReady
	}
	if o.RetryDelay <= 0 {
		o.RetryDelay = defaultRetry
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.spawn == nil {
		o.spawn = spawnChild
	}
	if o.probe == nil {
		o.probe = probeHTTP
	}
	if o.procCmd == nil {
		o.procCmd = procCmdline
	}
	if o.contenders == nil {
		o.contenders = doctor.Contenders
	}
	if o.probeGPU == nil {
		o.probeGPU = doctor.Probe
	}
}

// loopbackHost reports whether the downstream bind is loopback (token
// optional per ARCHITECTURE §7 auth).
func loopbackHost(host string) bool {
	return host == "localhost" || host == "::1" || host == "127.0.0.1" ||
		strings.HasPrefix(host, "127.")
}

// probeDownstream inspects the configured port: ours (healthz marker),
// listening-foreign, or free.
func probeDownstream(ctx context.Context, host string, port int) (ours, listening bool, err error) {
	addr := net.JoinHostPort(host, strconv.Itoa(port))
	conn, derr := net.DialTimeout("tcp", addr, 500*time.Millisecond)
	if derr != nil {
		return false, false, nil // nothing usable there → free
	}
	conn.Close()
	rctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, "http://"+addr+healthzPath, nil)
	if err != nil {
		return false, true, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, true, nil // something listens but does not answer our probe
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK && resp.Header.Get(markerHeader) == "1", true, nil
}

// freePort asks the kernel for a free loopback port (child binds :0
// semantics without touching TabbyAPI's config format). ponytail: tiny
// allocate-then-close race on loopback — revisit only if clashes appear.
func freePort() (int, error) {
	ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", "0"))
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

func generateToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func loadUpstreamKey(path string) string {
	if path == "" {
		return ""
	}
	b, err := os.ReadFile(path) // missing file → no injection (keyless upstreams)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// daemon is the live Serve state. Only the goroutines spawned by Serve
// touch it after startup, guarded by mu (status/load handlers race the
// sampler and supervisor).
type daemon struct {
	opts      Options
	conn      Conn // THE backend contract (see mux)
	token     string
	needToken bool
	upKey     string // upstream Bearer read from conn.UpKeyFile (file-only)
	port      int    // internal child port; 0 in attach mode
	cfgPath   string // generated child config (supervised only)
	logPath   string // logs/tabby.log (child stdout+stderr)
	child     *child // current incarnation (supervised only)

	mu     sync.Mutex
	loadMu sync.Mutex // serializes /-/load and /-/unload
	st     State
	fit    *autofit.Result // last accepted fit for OOM advice (nil = unknown)
}

func (d *daemon) setChild(c *child) {
	d.mu.Lock()
	d.child = c
	pid := 0
	if c != nil && c.cmd != nil && c.cmd.Process != nil {
		pid = c.cmd.Process.Pid
	}
	d.st.ChildPID = pid
	err := WriteState(d.opts.DataDir, d.st)
	d.mu.Unlock()
	if err != nil {
		fmt.Fprintf(d.opts.Stderr, "stone-llama: persist state: %v\n", err)
	}
}

func (d *daemon) childPID() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.st.ChildPID
}

// modelLoaded reports whether a model is loaded in state (supervise's
// fail-fast check after a crash).
func (d *daemon) modelLoaded() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.st.Model != ""
}

// probeAttach checks contract item 2 against an arbitrary base URL:
// GET {base}/v1/models; any HTTP answer within the bound = serving.
func probeAttach(ctx context.Context, base string, timeout time.Duration) error {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(cctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return nil
}

// classifyCrash turns a child exit into the error a user should act on:
// a foreign VRAM holder (with nvidia-smi evidence) is fatal and named;
// a load/prefill OOM is fatal and names the levers; anything else is
// restartable (nil).
func (d *daemon) classifyCrash(childErr error) error {
	tail := logTail(d.logPath, 20)
	if ce := d.contention(); ce != nil {
		return fmt.Errorf("serve: %s exited: %v; %w%s", d.conn.errName(), childErr, ce, tail)
	}
	if isOOM([]byte(tail)) {
		return fmt.Errorf("serve: %s exited: %v\n%s\n%s", d.conn.errName(), childErr, tail, d.oomAdvice())
	}
	return nil
}

// contention reports the foreign processes holding GPU compute memory
// (our own child excluded) as an explicit error, or nil when nothing
// foreign holds it or the evidence cannot be gathered.
func (d *daemon) contention() error {
	cs, err := d.opts.contenders()
	if err != nil {
		return nil // no evidence → make no claim
	}
	pid := d.childPID()
	var foreign []doctor.Contender
	for _, c := range cs {
		if c.PID != pid {
			foreign = append(foreign, c)
		}
	}
	return doctor.ContentionError(foreign)
}

// oomAdvice names the levers after an out-of-memory failure, with the
// numbers this daemon actually used (live evidence 2026-09-26: OOM
// happens during prefill before the first token; max_tokens=1 does not
// help because allocation precedes generation).
func (d *daemon) oomAdvice() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.opts.Attach != "" {
		return "stone-llama does not own this upstream (attach mode): lower --ctx/--cache-mode on the server that loads the model"
	}
	if d.fit == nil {
		return "lower --ctx or pick a stronger --cache-mode, then retry the load"
	}
	f := *d.fit
	return fmt.Sprintf("out of memory — lower --ctx (currently %d) or use a stronger --cache-mode (currently %s); largest ctx that would fit: %d",
		f.Ctx, f.Mode, f.LargestCtx)
}

// Serve runs the daemon until ctx is cancelled. Clean shutdown returns
// nil; everything else is an error the CLI prints.
func Serve(ctx context.Context, opts Options) error {
	opts.withDefaults()
	if opts.Port == 0 { // ephemeral: pick a free downstream port
		p, perr := freePort()
		if perr != nil {
			return fmt.Errorf("serve: pick downstream port: %w", perr)
		}
		opts.Port = p
	}

	ours, listening, err := probeDownstream(ctx, opts.Host, opts.Port)
	if err != nil {
		return err
	}
	if ours {
		return ErrAlreadyServing
	}
	if listening {
		return ErrPortInUse
	}

	lock, err := fslock.TryAcquire(LockPath(opts.DataDir))
	if err != nil {
		if errors.Is(err, fslock.ErrBusy) {
			if ours, _, perr := probeDownstream(ctx, opts.Host, opts.Port); perr == nil && ours {
				return ErrAlreadyServing
			}
			return ErrStartBusy
		}
		return err
	}
	defer lock.Release()

	// upstream: attach URL, or a supervised child. A schemeless
	// host:port (the documented form) defaults to http://.
	upstream := opts.Attach
	if upstream != "" && !strings.Contains(upstream, "://") {
		upstream = "http://" + upstream
	}
	if upstream != "" {
		u, perr := url.Parse(upstream)
		if perr != nil || (u.Scheme != "http" && u.Scheme != "https") {
			return fmt.Errorf("serve: --attach must be host:port or an http(s) URL, got %q", upstream)
		}
		if u.User != nil {
			return errors.New("serve: credentials in --attach URLs are not allowed — use the key file")
		}
		upstream = strings.TrimRight(u.String(), "/")
		opts.Attach = upstream
		if err := probeAttach(ctx, upstream, 10*time.Second); err != nil {
			return fmt.Errorf("serve: attach target %s not reachable: %w", upstream, err)
		}
	}

	token, err := generateToken()
	if err != nil {
		return err
	}
	d := &daemon{
		opts:      opts,
		token:     token,
		needToken: !loopbackHost(opts.Host),
		upKey:     loadUpstreamKey(opts.UpstreamKeyFile),
		logPath:   filepath.Join(opts.DataDir, "logs", "tabby.log"),
		st: State{
			PID:       os.Getpid(),
			Host:      opts.Host,
			Port:      opts.Port,
			Token:     token,
			Attach:    opts.Attach,
			StartedAt: opts.Now().Unix(),
		},
	}
	if opts.Attach != "" {
		d.conn = Conn{BaseURL: upstream, UpKeyFile: opts.UpstreamKeyFile}
	}

	supCtx, supCancel := context.WithCancel(ctx)
	defer supCancel()

	var c *child
	if opts.Attach == "" {
		port, perr := freePort()
		if perr != nil {
			return perr
		}
		d.port = port
		d.cfgPath = tabbyCfgPath(opts.RuntimeDir)
		// Backend keys pre-created in the child's CWD (it reads ours
		// instead of generating/logging its own) plus the raw upstream
		// key file the proxy injects from — both 0600, never argv/log.
		keyFile, _, terr := writeChildTokens(opts.RuntimeDir)
		if terr != nil {
			return fmt.Errorf("serve: write child keys: %w", terr)
		}
		d.upKey = loadUpstreamKey(keyFile)
		d.conn = Conn{
			BaseURL:   "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)),
			Name:      tabbyConnName(opts.PinnedCommit),
			UpKeyFile: keyFile,
		}
		yaml := RenderTabbyYAML(TabbyConfig{
			Host:     "127.0.0.1",
			Port:     port,
			ModelDir: opts.ModelsDir,
			// model-less at boot: /-/load drives the backend's native
			// load endpoint (no child restart on model switch).
		})
		if werr := writeSecret(d.cfgPath, yaml); werr != nil {
			return fmt.Errorf("serve: write child config: %w", werr)
		}
		c, err = d.startReady(supCtx)
		if err != nil {
			return err
		}
	}

	// Listen first so an ephemeral port (:0) is adopted into state
	// before any client can dial it; daemon.json lands only for a live
	// listener.
	ln, lerr := net.Listen("tcp", d.st.Addr())
	if lerr != nil {
		if c != nil {
			c.shutdownChild(killGrace)
		}
		return fmt.Errorf("serve: listen %s: %w", d.st.Addr(), lerr)
	}
	if err := WriteState(opts.DataDir, d.st); err != nil {
		ln.Close()
		if c != nil {
			c.shutdownChild(killGrace)
		}
		return err
	}

	srv := &http.Server{Handler: d.authed(d.mux())}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	supErr := make(chan error, 1)
	if c != nil {
		go func() { supErr <- d.supervise(supCtx, c) }()
	}
	// attach has no supervisor: a nil channel blocks forever in select,
	// so an empty supErr must never be pre-filled (it would read as an
	// immediate "daemon exited nil").
	var supCh <-chan error = supErr
	if c == nil {
		supCh = nil
	}
	go d.sampleVRAM(supCtx)

	fmt.Fprintf(opts.Stdout, "stone-llama listening on %s (OpenAI-compatible)\n", d.st.Addr())
	if opts.Attach != "" {
		fmt.Fprintf(opts.Stdout, "  upstream: %s (attach)\n", upstream)
	} else {
		fmt.Fprintf(opts.Stdout, "  runtime: %s (internal port hidden)\n", d.conn.Name)
	}
	if d.needToken {
		fmt.Fprintf(opts.Stdout, "  auth: bearer token required (token file: %s)\n", StatePath(opts.DataDir))
	}

	var result error
	consumed := false
	select {
	case <-ctx.Done(): // clean shutdown (stop / SIGTERM)
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			result = fmt.Errorf("serve: %w", err)
		}
	case err := <-supCh:
		consumed = true
		result = err
	}
	if !consumed {
		supCancel()
		if c != nil {
			if err := <-supErr; err != nil && result == nil {
				result = err
			}
		}
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	srv.Shutdown(shutdownCtx)
	cancel()
	RemoveState(opts.DataDir)
	return result
}

// startReady spawns the first child incarnation and waits for
// readiness, restarting bounded times on exit with backoff; a child
// that never becomes ready fails with log forensics (classified for
// GPU contention / OOM first).
func (d *daemon) startReady(ctx context.Context) (*child, error) {
	for attempt := 1; ; attempt++ {
		c, err := d.spawn()
		if err != nil {
			return nil, err
		}
		d.setChild(c)
		died, rerr := awaitReady(ctx, c, d.opts.ReadyTimeout, d.opts.RetryDelay, d.opts.probe)
		if !died && rerr == nil {
			return c, nil
		}
		if died {
			if rerr == nil {
				// cmd.Wait() is nil on a clean exit — name it so the
				// restart/fail forensics don't print "(<nil>)".
				rerr = errCleanExit
			}
			// awaitReady consumed this incarnation's exit (H1): push it
			// back so every c.wait consumer (supervise, shutdownChild)
			// still gets exactly one value — without this a child that
			// exited 0 before readiness hung shutdownChild forever on
			// its receive and left supervise waiting for an exit that
			// had already been read.
			c.wait <- rerr
		}
		if ctx.Err() != nil {
			c.shutdownChild(killGrace)
			return nil, ctx.Err()
		}
		if !died {
			c.shutdownChild(killGrace)
			if fatal := d.classifyCrash(rerr); fatal != nil {
				return nil, fatal
			}
			return nil, fmt.Errorf("serve: %w%s", rerr, logTail(d.logPath, 20))
		}
		if fatal := d.classifyCrash(rerr); fatal != nil {
			return nil, fatal
		}
		if attempt >= childRestarts {
			return nil, fmt.Errorf("serve: %s exited %d times (%v)%s", d.conn.errName(), attempt, rerr, logTail(d.logPath, 20))
		}
		fmt.Fprintf(d.opts.Stderr, "stone-llama: %s exited (%v), restarting (%d/%d)\n",
			d.conn.errName(), rerr, attempt, childRestarts)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(attempt) * d.opts.RetryDelay):
		}
	}
}

func (d *daemon) spawn() (*child, error) {
	return d.opts.spawn(d.opts.RuntimeDir, d.cfgPath, d.logPath, d.port)
}

// supervise owns the wait channel of every incarnation after the first
// ready. A crash while NO model is loaded restarts bounded times with
// backoff and log forensics (boot/config storms); a crash AFTER a model
// was loaded fails fast — the model is gone, and silently serving
// without it would lie. Contention/OOM are classified first and always
// fatal, with evidence. Returns nil on ctx shutdown (child reaped).
func (d *daemon) supervise(ctx context.Context, c *child) error {
	attempt := 1
	var pending error // exit already received from c.wait
	for {
		var exitErr error
		if pending != nil {
			exitErr, pending = pending, nil
		} else {
			select {
			case <-ctx.Done():
				c.shutdownChild(killGrace)
				return nil
			case exitErr = <-c.wait:
			}
		}
		name := d.conn.errName()
		if fatal := d.classifyCrash(exitErr); fatal != nil {
			return fatal
		}
		if d.modelLoaded() {
			return fmt.Errorf("serve: %s crashed while a model was loaded: %v%s",
				name, exitErr, logTail(d.logPath, 20))
		}
		if attempt >= childRestarts {
			return fmt.Errorf("serve: %s exited %d times (%v)%s",
				name, attempt, exitErr, logTail(d.logPath, 20))
		}
		fmt.Fprintf(d.opts.Stderr, "stone-llama: %s exited (%v), restarting (%d/%d)\n",
			name, exitErr, attempt, childRestarts)
		select {
		case <-ctx.Done():
			return nil // child already exited; nothing left to reap
		case <-time.After(time.Duration(attempt) * d.opts.RetryDelay):
		}
		nc, err := d.spawn()
		if err != nil {
			return fmt.Errorf("serve: restart: %w%s", err, logTail(d.logPath, 20))
		}
		d.setChild(nc)
		died, rerr := awaitReady(ctx, nc, d.opts.ReadyTimeout, d.opts.RetryDelay, d.opts.probe)
		if rerr != nil && !died {
			nc.shutdownChild(killGrace)
			if ctx.Err() != nil {
				return nil
			}
			if fatal := d.classifyCrash(rerr); fatal != nil {
				return fatal
			}
			return fmt.Errorf("serve: %s not ready after restart: %v%s", name, rerr, logTail(d.logPath, 20))
		}
		if died {
			// awaitReady consumed the exit — hand it back so this loop
			// stays the single receiver (buffered chan, slot is free).
			nc.wait <- rerr
		}
		c = nc
		attempt++
	}
}

// sampleVRAM tracks the child's GPU memory for ps's VRAM PEAK column.
// No child (attach) or no nvidia-smi → skips silently; ps then shows -.
func (d *daemon) sampleVRAM(ctx context.Context) {
	t := time.NewTicker(vramSampleEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		pid := d.childPID()
		if pid == 0 {
			continue
		}
		cs, err := d.opts.contenders()
		if err != nil {
			continue
		}
		peak := 0
		for _, c := range cs {
			if c.PID == pid {
				peak = int(c.MiB)
				break
			}
		}
		if peak <= 0 {
			continue
		}
		d.mu.Lock()
		if peak > d.st.VRAMPeakMiB {
			d.st.VRAMPeakMiB = peak
			werr := WriteState(d.opts.DataDir, d.st)
			d.mu.Unlock()
			if werr != nil {
				fmt.Fprintf(d.opts.Stderr, "stone-llama: persist state: %v\n", werr)
			}
			continue
		}
		d.mu.Unlock()
	}
}

// Conn names the backend contract the proxy depends on — exactly three
// things, nothing more:
//
//  1. an http.Handler at BaseURL speaking the OpenAI-compatible surface;
//  2. GET {BaseURL}/v1/models answers within ReadyTimeout — any status
//     counts as ready ("serving", not "loaded");
//  3. BaseURL streams SSE tokens with the proxy's FlushInterval -1
//     (zero-buffer).
//
// Attach mode supplies exactly these three and no Backend; the
// supervised backend (tabby_backend.go + tabbyconf.go) additionally
// owns spawn/readiness/shutdown, the YAML config schema, start.py
// argv, the api_tokens.yml name and auth scheme, and the probe detail —
// none of which may leak past those files. Every user-facing backend
// mention routes through Conn.Name.
type Conn struct {
	BaseURL   string // proxy target, e.g. http://127.0.0.1:41337
	Name      string // display name (banner); "" → errors stay generic
	UpKeyFile string // file whose contents become the upstream Bearer
}

// errName is how errors refer to the backend: the display name when we
// have one, otherwise the generic "inference backend".
func (c Conn) errName() string {
	if c.Name != "" {
		return c.Name
	}
	return "inference backend"
}

// mux routes our control endpoints plus the OpenAI-compatible proxy.
func (d *daemon) mux() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(healthzPath, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(markerHeader, "1")
		io.WriteString(w, healthzBody)
	})
	mux.HandleFunc(statusPath, d.handleStatus)
	mux.HandleFunc(loadPath, d.handleLoad)
	mux.HandleFunc(unloadPath, d.handleUnload)

	target, _ := url.Parse(d.conn.BaseURL)
	rp := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme = target.Scheme
			r.URL.Host = target.Host
			r.URL.Path = joinPath(target.Path, r.URL.Path)
			r.Host = target.Host
			// downstream auth never crosses; upstream key injected from
			// memory (written from file/generated, never argv/log).
			r.Header.Del("Authorization")
			if d.upKey != "" {
				r.Header.Set("Authorization", "Bearer "+d.upKey)
			}
		},
		// zero-buffer SSE: flush after every write, no response
		// buffering, no content-length meddling (ARCHITECTURE §7).
		FlushInterval:  -1,
		ModifyResponse: d.rewriteOOM,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, e error) {
			fmt.Fprintf(d.opts.Stderr, "stone-llama: proxy %s: %v\n", r.URL.Path, e)
			writeJSON(w, http.StatusBadGateway, "upstream unavailable: "+e.Error(), "proxy_error")
		},
	}
	mux.Handle("/", rp)
	return mux
}

// authed enforces downstream bearer: healthz open (our own probe), token
// required when host is not loopback, anything accepted on loopback
// (optional per §7) — and whatever arrives is stripped by the Director.
func (d *daemon) authed(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == healthzPath {
			next.ServeHTTP(w, r)
			return
		}
		if d.needToken {
			got := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer"))
			got = strings.TrimSpace(got)
			if got == "" || subtle.ConstantTimeCompare([]byte(got), []byte(d.token)) != 1 {
				w.Header().Set("WWW-Authenticate", "Bearer")
				writeJSON(w, http.StatusUnauthorized, "bearer token required", "unauthorized")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// joinPath prefixes the upstream base path (attach URLs may carry one)
// onto the client path, both slash-normalized.
func joinPath(prefix, p string) string {
	if prefix == "" || prefix == "/" {
		return p
	}
	if p == "" || p == "/" {
		return strings.TrimRight(prefix, "/")
	}
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return strings.TrimRight(prefix, "/") + p
}

// daemonCmdline is the Stop identification gate: argv0 names
// stone-llama and argv1 is `serve` — exactly how StartDetached execs
// the daemon. The old check was a bare "stone-llama" substring: it
// matched unrelated processes before the signal, and its wait-loop
// twin matched nothing at all (empty cmdline never contains the
// substring), so a recycled pid's killer could never confirm the exit
// (H2). Positional argv1 also keeps `stone-llama run <model named
// serve>` out of the gate.
func daemonCmdline(cmd string) bool {
	f := strings.Fields(cmd)
	return len(f) >= 2 && strings.Contains(f[0], "stone-llama") && f[1] == "serve"
}

// Stop terminates the daemon recorded in state. An absent state is a
// no-op; a recycled pid (no longer a stone-llama serve) drops stale
// state without signalling. Identification is two-gated (H2): the
// cmdline must read as our serve AND the recorded port must answer the
// healthz marker — if it will not confirm itself, we refuse to signal
// and keep the state for inspection.
func Stop(dataDir string, grace time.Duration) error {
	if runtime.GOOS == "windows" {
		return errors.New("stop is not supported on windows (v1 targets Linux; the windows build compiles but is untested)")
	}
	st, err := ReadState(dataDir)
	if err != nil {
		if errors.Is(err, ErrNoDaemon) {
			return nil
		}
		if !errors.Is(err, ErrCorruptState) || !st.usable() {
			return err // corrupt and unusable: report clearly, keep the file
		}
		// corrupt but salvaged pid/port (H3): fall through — the
		// cmdline and marker gates below decide whether anything is
		// signalled.
	}
	rtDir := runtimeDirFor(dataDir)
	if !daemonCmdline(procCmdline(st.PID)) {
		// pid recycled by an unrelated process — never signal it
		return dropState(dataDir, rtDir)
	}
	ours, listening, perr := probeDownstream(context.Background(), st.Host, st.Port)
	if perr != nil || !ours {
		return fmt.Errorf("pid %d reads as a stone-llama serve, but %s does not answer the healthz marker (listening=%v) — refusing to signal a process that is not confirmed ours; state kept at %s",
			st.PID, st.Addr(), listening, StatePath(dataDir))
	}
	proc, err := os.FindProcess(st.PID)
	if err != nil {
		return dropState(dataDir, rtDir)
	}
	if err := proc.Signal(syscall.SIGTERM); err != nil {
		if errors.Is(err, os.ErrProcessDone) {
			return dropState(dataDir, rtDir)
		}
		return err
	}
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		if !daemonCmdline(procCmdline(st.PID)) {
			return dropState(dataDir, rtDir)
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("serve: pid %d did not exit within %s", st.PID, grace)
}

// dropState ends a daemon's footprint: daemon.json plus the files this
// binary generates under the data dir. Ownership: generated by a
// stone-llama serve instance, removed exactly when its state goes away
// (clean shutdown, `stop`, failed start) — see cleanupGenerated.
func dropState(dataDir, runtimeDir string) error {
	err := RemoveState(dataDir)
	cleanupGenerated(dataDir, runtimeDir)
	return err
}

// cleanupGenerated removes the per-daemon secrets and the coordination
// lock — allow-listed exact paths only, never the runtime wholesale,
// never recursive. Symlink discipline (venv-adoption safety): every
// path is checked with Lstat — never followed — and every component
// below root must be a real path; a symlink anywhere in the chain
// (runtime/venv → a user's pre-existing virtualenv, or tabbyAPI → an
// adopted checkout) means the path reaches into a user-owned
// environment and is left completely alone. runtime/venv is not in
// the allow-list at all: provisioned venvs (~5.6 GB) and adopted
// symlink targets both belong to the user. Attach mode never writes
// these files (writeChildTokens is supervised-only), so it never
// persists a key either.
func cleanupGenerated(dataDir, runtimeDir string) {
	removeOwn(filepath.Join(runtimeDir, "upstream_key"), runtimeDir)
	removeOwn(tabbyCfgPath(runtimeDir), runtimeDir)
	removeOwn(filepath.Join(runtimeDir, "tabbyAPI", "api_tokens.yml"), runtimeDir)
	removeOwnDir(filepath.Join(runtimeDir, "tabbyAPI"), runtimeDir)
	// ponytail: unlinking the flock file can race a concurrent
	// TryAcquire onto a fresh inode (microsecond double-hold window at
	// teardown only); the port guard (ErrAlreadyServing) still blocks a
	// second daemon. Revisit with inode-stamped locks if a download
	// ever slips through that window.
	removeOwn(LockPath(dataDir), dataDir)
}

// ownPath reports path is ours to unlink: under root, and every
// component below root exists and is NOT a symlink (os.Lstat, never a
// following lookup). One check covers both dangers: a link at the slot
// itself, and a link in a parent component (deleting through it would
// reach into the linked environment).
func ownPath(path, root string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false // not under root: not ours to decide
	}
	cur := root
	for _, part := range strings.Split(rel, string(filepath.Separator)) {
		cur = filepath.Join(cur, part)
		fi, lerr := os.Lstat(cur)
		if lerr != nil || fi.Mode()&os.ModeSymlink != 0 {
			return false // absent, unreadable, or a link into a user env
		}
	}
	return true
}

// removeOwn unlinks path only when ownPath passes; absent is fine.
func removeOwn(path, root string) {
	if ownPath(path, root) {
		_ = os.Remove(path)
	}
}

// removeOwnDir unlinks path only when it is a real directory (symlinks
// already rejected by ownPath) and empty — os.Remove refuses a
// non-empty dir, so an adopted or checked-out tabbyAPI never loses
// files.
func removeOwnDir(path, root string) {
	if !ownPath(path, root) {
		return
	}
	if fi, err := os.Lstat(path); err != nil || !fi.IsDir() {
		return
	}
	_ = os.Remove(path)
}

// runtimeDirFor resolves the generated-files dir for state-level
// operations (Stop runs without Options): configured runtime dir, else
// the data-dir default. A broken config falls back to the default —
// cleanup must never guess wide.
func runtimeDirFor(dataDir string) string {
	if cfg, err := config.Load(); err == nil && cfg.RuntimeDir != "" {
		return cfg.RuntimeDir
	}
	return filepath.Join(dataDir, "runtime")
}

// LoadedModel reports the model the running daemon has loaded, if any
// (rm refuses to delete it). Attach mode never reports: the upstream
// owns its lifecycle and its model is not ours to guard.
func LoadedModel(dataDir string) (string, bool) {
	st, err := ReadState(dataDir)
	if err != nil || st.Model == "" || st.Attach != "" {
		return "", false
	}
	return st.Model, true
}

// EnsureDaemon returns a live daemon state. With start, a missing or
// dead daemon is auto-started detached (re-exec of this binary,
// `serve`, output → logs/daemon.log); the starter exits when the
// daemon fails, which surfaces THIS attempt's log output instead of a
// hang. start=false is the STONE_LLAMA_NO_AUTOSTART=1 path: it reports
// what was actually found — never a bare "not running" while a
// listener holds the port (A3) — and never deletes state (H3): one
// failed probe is not proof of death; state is only removed by the
// daemon's own shutdown or `stone-llama stop`.
func EnsureDaemon(dataDir string, start bool, timeout time.Duration) (State, error) {
	// A live listener returns here — including corrupt-but-salvaged
	// state, so ps/stop reach a running daemon instead of hiding it
	// behind the parse error (H3).
	st, rerr := ReadState(dataDir)
	switch {
	case rerr == nil, errors.Is(rerr, ErrCorruptState) && st.usable():
		if class := classifyListener(context.Background(), st); class == probeLive {
			return st, nil
		} else if !start {
			return State{}, stateDownErr(dataDir, st, class, rerr)
		}
		// start=true: keep the state (a booting daemon may own the
		// port); a successful start overwrites it, a failure reports
		// what it found. Nothing here deletes (H3).
	case errors.Is(rerr, ErrCorruptState):
		if !start {
			return State{}, fmt.Errorf("%v — no usable pid/port salvaged, kept at %s for repair (not deleted)", rerr, StatePath(dataDir))
		}
		// start=true: fall through; a successful start rewrites the file.
	default: // no state (ErrNoDaemon) or a read error
		if !start {
			return State{}, missingStateErr(dataDir, rerr)
		}
	}
	alive := func() (State, bool) {
		st, err := ReadState(dataDir)
		if err != nil && !(errors.Is(err, ErrCorruptState) && st.usable()) {
			return State{}, false
		}
		return st, classifyListener(context.Background(), st) == probeLive
	}
	exe, err := os.Executable()
	if err != nil {
		return State{}, fmt.Errorf("locate executable: %w", err)
	}
	// Scope the failure tail to bytes THIS attempt appends: the log on
	// disk holds every past attempt too, and re-printing it turned one
	// failure into a wall of duplicates (A5).
	logPath := filepath.Join(dataDir, "logs", "daemon.log")
	var logStart int64
	if fi, serr := os.Stat(logPath); serr == nil {
		logStart = fi.Size()
	}
	done, err := startDetached(exe, dataDir)
	if err != nil {
		return State{}, err
	}
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	for {
		select {
		case <-done:
			// starter exited: either the daemon came up (race with a
			// concurrent starter) or it failed — say why, from what
			// this attempt found (port first, else the scoped tail).
			if st, ok := alive(); ok {
				return st, nil
			}
			return State{}, startFailed(dataDir, logStart)
		case <-tick.C:
			if st, ok := alive(); ok {
				return st, nil
			}
		case <-deadline.C:
			return State{}, fmt.Errorf("daemon still starting after %s — check %s", timeout, logPath)
		}
	}
}

// startDetached is the auto-start seam: tests inject a fake starter so
// failure paths never re-exec the test binary.
var startDetached = StartDetached

// stateDownErr states exactly what answered a recorded port — nothing,
// a foreign process, or a listener that would not confirm itself (A3),
// and never deletes the state (H3).
func stateDownErr(dataDir string, st State, class probeClass, stateErr error) error {
	note := ""
	if stateErr != nil {
		note = fmt.Sprintf(" [state file: %v]", stateErr)
	}
	switch class {
	case probeUnconfirmed:
		return fmt.Errorf("could not confirm a stone-llama daemon on %s: a listener answers neither the healthz marker nor /-/status with the state token (booting or wedged?)%s — state kept at %s", st.Addr(), note, StatePath(dataDir))
	case probeForeign:
		return fmt.Errorf("a foreign process holds %s (not a stone-llama daemon) and the recorded state no longer matches it%s — state kept at %s; 'stone-llama stop' clears it", st.Addr(), note, StatePath(dataDir))
	default: // probeDown
		return fmt.Errorf("no stone-llama daemon is listening on %s%s — stale state kept at %s; 'stone-llama stop' clears it", st.Addr(), note, StatePath(dataDir))
	}
}

// missingStateErr reports what was found when no state file exists:
// never a bare "not running" while a listener holds the configured
// port (A3) — name the holder and where state would have lived.
func missingStateErr(dataDir string, rerr error) error {
	if rerr != nil && !errors.Is(rerr, ErrNoDaemon) {
		return fmt.Errorf("read %s: %w", StatePath(dataDir), rerr)
	}
	if cfg, cerr := config.Load(); cerr == nil {
		cfgSt := State{Host: cfg.Host, Port: cfg.Port}
		switch classifyListener(context.Background(), cfgSt) {
		case probeLive:
			return fmt.Errorf("a stone-llama daemon is already serving on %s, but its state is not in %s (state file missing or another data dir) — not starting a second one", cfgSt.Addr(), dataDir)
		case probeForeign:
			return fmt.Errorf("another process holds %s (not a stone-llama daemon) and %s has no state file — nothing to query from here", cfgSt.Addr(), dataDir)
		case probeUnconfirmed:
			return fmt.Errorf("a listener on %s would not identify itself (booting or wedged?) and %s has no state file — nothing to query from here", cfgSt.Addr(), dataDir)
		}
	}
	return fmt.Errorf("no stone-llama daemon is running (start one with 'stone-llama serve')")
}

// startFailed diagnoses one failed auto-start: what the configured
// port answers now (competition, foreign holder, boot window), else
// this attempt's own log lines — A5-scoped, so history never repeats.
func startFailed(dataDir string, logStart int64) error {
	if cfg, cerr := config.Load(); cerr == nil {
		cfgSt := State{Host: cfg.Host, Port: cfg.Port}
		switch classifyListener(context.Background(), cfgSt) {
		case probeLive:
			return fmt.Errorf("a stone-llama daemon is already serving on %s, but its state is not in %s (state file missing or another data dir) — not starting a second one", cfgSt.Addr(), dataDir)
		case probeForeign:
			return fmt.Errorf("another process holds %s (not a stone-llama daemon) — stop it, or pick a free port with --port", cfgSt.Addr())
		case probeUnconfirmed:
			return fmt.Errorf("a listener on %s answered neither the healthz marker nor /-/status (a daemon may be booting or wedged) — retry shortly; state kept at %s", cfgSt.Addr(), StatePath(dataDir))
		}
	}
	logPath := filepath.Join(dataDir, "logs", "daemon.log")
	detail := logTailSince(logPath, logStart, 10)
	if detail == "" {
		return fmt.Errorf("auto-start failed: starter produced no output (log: %s)", logPath)
	}
	return fmt.Errorf("auto-start failed:%s", detail)
}

// probeClass distinguishes what answers a recorded port: our daemon,
// nothing, a foreign process, or a listener that confirms neither
// (booting/stalled — one failed probe is never proof of death, H3).
type probeClass int

const (
	probeUnknown     probeClass = iota // no usable state to probe
	probeLive                          // healthz marker, or /-/status with the state token
	probeDown                          // nothing listening
	probeUnconfirmed                   // listener up, neither probe answered within bound
	probeForeign                       // listener answered without our identity
)

// classifyListener identifies st's port: the healthz marker is the
// primary proof; /-/status carrying the unguessable state token is the
// cross-check when healthz was inconclusive (H3: identification must
// separate confirmed-live, confirmed-foreign, and could-not-confirm).
func classifyListener(ctx context.Context, st State) probeClass {
	conn, derr := net.DialTimeout("tcp", st.Addr(), 500*time.Millisecond)
	if derr != nil {
		return probeDown
	}
	conn.Close()
	rctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, "http://"+st.Addr()+healthzPath, nil)
	if err != nil {
		return probeUnconfirmed
	}
	resp, err := http.DefaultClient.Do(req)
	if err == nil {
		mine := resp.StatusCode == http.StatusOK && resp.Header.Get(markerHeader) == "1"
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if mine {
			return probeLive
		}
	}
	answered, ok := statusAnswers(ctx, st)
	if ok {
		return probeLive
	}
	if answered {
		return probeForeign
	}
	return probeUnconfirmed
}

// statusAnswers cross-checks /-/status with the recorded token:
// (answered, ok). ok proves our daemon (token unguessable); answered
// with !ok proves a foreign HTTP server; no answer at all is a
// booting/stalled listener — not yet dead.
func statusAnswers(ctx context.Context, st State) (answered, ok bool) {
	rctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, "http://"+st.Addr()+statusPath, nil)
	if err != nil {
		return false, false
	}
	if st.Token != "" {
		req.Header.Set("Authorization", "Bearer "+st.Token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return true, resp.StatusCode == http.StatusOK
}

// writeSecret writes a 0600 file atomically (child config/tokens —
// same discipline as daemon.json).
func writeSecret(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// execOutput runs a command for platform helpers (procCmdline fallback).
func execOutput(name string, args ...string) string {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return ""
	}
	return string(out)
}
