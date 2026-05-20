PYTHON  ?= .venv/bin/python
PYTEST  ?= .venv/bin/pytest

VERSION            ?= 0.1.0-dev
CONTAINER_REGISTRY ?= ghcr.io/soliddowant
PUSH_ALL           ?= false

INCLUDE_LATEST = $(PUSH_ALL)

IMAGE_NAME = bazarr-whisper-proxy
IMAGE_TAGS = $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(VERSION) \
             $(if $(filter true,$(INCLUDE_LATEST)),$(CONTAINER_REGISTRY)/$(IMAGE_NAME):latest)

.PHONY: help
help:  ## List available targets.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

.PHONY: lint
lint:  ## Check code style (ruff check + format --check).
	ruff check .
	ruff format --check .

.PHONY: fmt
fmt:  ## Auto-fix style issues (ruff format + check --fix).
	ruff format .
	ruff check --fix .

.PHONY: test
test:  ## Run unit/integration tests (excludes e2e).
	$(PYTEST) tests/ --ignore=tests/e2e

.PHONY: e2e
e2e:  ## Run the end-to-end suite. Requires OPENARC_E2E_BASE_URL. Pass ARGS=--keep-up to leave the stack running.
	scripts/e2e.sh $(ARGS)

.PHONY: build-image
build-image:  ## Build the OCI image via Nix and load it into the local Docker daemon.
	$$(nix --extra-experimental-features 'nix-command flakes' build --print-out-paths --no-link .#dockerImage) | docker load
	$(foreach tag,$(IMAGE_TAGS),docker tag $(IMAGE_NAME):latest $(tag);)
	$(if $(filter true,$(PUSH_ALL)),$(foreach tag,$(IMAGE_TAGS),docker push $(tag);))

.PHONY: release
release: TAG           = v$(VERSION)
release: SAFETY_PREFIX = $(if $(filter true,$(PUSH_ALL)),,echo)
release: build-image ## Create a GitHub release. Set PUSH_ALL=true to tag, push, and publish. Requires the GitHub CLI (gh).
	@gh auth status
	@$(SAFETY_PREFIX) git tag -a $(TAG) -m "Release $(TAG)"
	@$(SAFETY_PREFIX) git push origin
	@$(SAFETY_PREFIX) git push origin --tags
	@$(SAFETY_PREFIX) gh release create $(TAG) --generate-notes --latest --verify-tag
