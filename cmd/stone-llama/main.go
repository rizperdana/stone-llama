package main

import (
	"os"

	"github.com/rizperdana/stone-llama/internal/cli"
)

// version is injected at release build time via
// -ldflags "-X main.version=vX.Y.Z"
var version = "dev"

func main() {
	os.Exit(cli.Run(os.Args[1:], version, os.Stdout, os.Stderr))
}
