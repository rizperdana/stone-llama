# Stone Llama — release engineering Makefile
# Static Go builds with cross-platform release artifacts.

BINARY := stone-llama
ENTRYPOINT := ./cmd/stone-llama

VERSION_TAG ?= $(shell git describe --tags --abbrev=0 2>/dev/null || echo dev)

LDFLAGS := -trimpath -ldflags "-s -w -X main.version=$(VERSION_TAG)"

# linux/amd64 is the only *supported* platform; the other four are published
# but compiled-only (see docs/RELEASE.md for the honest platform matrix).
# linux/arm64 is also untested here — no arm64 machine.
RELEASE_TARGETS := linux/amd64 linux/arm64 windows/amd64 darwin/amd64 darwin/arm64

# sha256 portability: Linux has sha256sum; macOS ships shasum -a 256
ifeq ($(shell command -v sha256sum >/dev/null 2>&1 && echo yes),yes)
	SHA256 := sha256sum
else
	SHA256 := shasum -a 256
endif

.PHONY: help build test fmt clean dist

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

build: ## Static binary for the current platform
	CGO_ENABLED=0 go build $(LDFLAGS) -o $(BINARY) $(ENTRYPOINT)

test: ## Run go vet and go test
	go vet ./...
	go test ./...

fmt: ## Format source with gofmt
	gofmt -s -w .

clean: ## Remove dist/ and local binary
	rm -rf dist/
	rm -f $(BINARY)

dist: clean ## Cross-compile release artifacts into dist/
	@mkdir -p dist/
	@fail=0; \
	for tgt in $(RELEASE_TARGETS); do \
		os=$${tgt%/*}; arch=$${tgt#*/}; \
		case $$os in \
			windows) bin=$(BINARY).exe; ext=zip ;; \
			*)       bin=$(BINARY);     ext=tgz ;; \
		esac; \
		name=$(BINARY)-$$os-$$arch; \
		stagedir=$$name; \
		rm -rf $$stagedir; mkdir -p $$stagedir; \
		echo "==> $$os/$$arch → $$name.$$ext"; \
		if ! CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build $(LDFLAGS) -o $$stagedir/$$bin $(ENTRYPOINT); then \
			echo "  BUILD FAILED for $$os/$$arch — skipping"; \
			fail=1; \
			rm -rf $$stagedir; \
			continue; \
		fi; \
		[ -f README.md ]  && cp README.md  $$stagedir/; \
		[ -f LICENSE ]    && cp LICENSE    $$stagedir/; \
		if [ -f assets/stone-llama.png ]; then cp assets/stone-llama.png $$stagedir/; fi; \
		if [ "$$ext" = "zip" ]; then \
			if command -v zip >/dev/null 2>&1; then \
				(cd $$stagedir && zip -qr ../dist/$$name.zip .); \
			else \
				(cd $$stagedir && ZNAME=$$name python3 -c 'import zipfile, os; z=zipfile.ZipFile("../dist/" + os.environ["ZNAME"] + ".zip", "w", zipfile.ZIP_DEFLATED); [z.write(os.path.join(r, f), os.path.join(os.environ["ZNAME"], os.path.relpath(os.path.join(r, f)))) for r, _, fs in os.walk(".") for f in fs]; z.close()'); \
			fi; \
		else \
			tar czf dist/$$name.tgz $$stagedir; \
		fi; \
		sz=$$(wc -c < dist/$$name.$$ext | tr -d ' '); \
		human=$$(du -h dist/$$name.$$ext | cut -f1); \
		echo "  → dist/$$name.$$ext ($$human, $$sz bytes)"; \
		rm -rf $$stagedir; \
	done; \
	(cd dist && for f in stone-llama-*.tgz stone-llama-*.zip; do [ -f "$$f" ] && $(SHA256) "$$f" || true; done > checksums.txt); \
	echo "==> Combined checksums: dist/checksums.txt"; \
	echo "==> dist/ contents:"; \
	ls -lh dist/; \
	if [ "$$fail" -ne 0 ]; then \
		echo ""; \
		echo "!!! Some targets failed — see errors above"; \
		echo "!!! dist/ contains artifacts for successful targets only"; \
		exit 1; \
	fi
