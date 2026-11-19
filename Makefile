MISE ?= mise
MISE_PRESENT := $(shell command -v $(MISE) >/dev/null 2>&1 && printf yes)
ifeq ($(MISE_PRESENT),yes)
GO ?= $(MISE) exec -- go
GOFMT ?= $(MISE) exec -- gofmt
else
GO ?= go
GOFMT ?= gofmt
endif
CONFIG ?= config.example.yaml
APP := ./cmd/ai-router
BUILD_FLAGS := -trimpath -buildvcs=true -ldflags="-s -w"

# Dashboard asset toolchain: templ generates Go from the .templ files, the
# standalone Tailwind CLI builds the stylesheet, Datastar is vendored as is.
DASHBOARD := internal/dashboard
TEMPL_VERSION := v0.3.1020
TAILWIND_VERSION := v4.3.3
DATASTAR_VERSION := v1.0.4
TAILWIND_PLATFORM ?= $(shell uname -s | sed 's/Darwin/macos/;s/Linux/linux/')-$(shell uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/')
TAILWIND := tools/tailwindcss-$(TAILWIND_VERSION)-$(TAILWIND_PLATFORM)

.PHONY: verify format-check vet race-test test build release-linux run check clean generate css datastar

verify: format-check vet race-test

format-check:
	@unformatted="$$(find . -type f -name '*.go' -not -path './.git/*' -print0 | xargs -0 $(GOFMT) -l)"; \
	if [ -n "$$unformatted" ]; then \
		printf 'Go files need gofmt:\n%s\n' "$$unformatted" >&2; \
		exit 1; \
	fi

vet:
	$(GO) vet ./...

race-test:
	$(GO) test -race ./...

test:
	$(GO) test ./...

build:
	mkdir -p dist
	CGO_ENABLED=0 $(GO) build $(BUILD_FLAGS) -o dist/ai-router $(APP)

release-linux:
	mkdir -p dist
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 $(GO) build $(BUILD_FLAGS) -o dist/ai-router-linux-amd64 $(APP)

run:
	$(GO) run $(APP) serve --config $(CONFIG)

check:
	$(GO) run $(APP) check --config $(CONFIG)

generate:
	$(GO) run github.com/a-h/templ/cmd/templ@$(TEMPL_VERSION) generate -path $(DASHBOARD)

css: $(TAILWIND)
	$(TAILWIND) --input $(DASHBOARD)/tailwind.css --output $(DASHBOARD)/static/dashboard.css --minify

$(TAILWIND):
	mkdir -p tools
	curl -fsSL -o $@ https://github.com/tailwindlabs/tailwindcss/releases/download/$(TAILWIND_VERSION)/tailwindcss-$(TAILWIND_PLATFORM)
	chmod +x $@

datastar:
	curl -fsSL -o $(DASHBOARD)/static/datastar.js https://raw.githubusercontent.com/starfederation/datastar/$(DATASTAR_VERSION)/bundles/datastar.js

clean:
	rm -rf dist
