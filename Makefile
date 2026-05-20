VERSION            ?= 0.1.0-dev
CONTAINER_REGISTRY ?= ghcr.io/soliddowant
PUSH_ALL           ?= false

INCLUDE_LATEST = $(PUSH_ALL)

IMAGE_NAME = bazarr-whisper-proxy
IMAGE_TAGS = $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(VERSION) \
             $(if $(filter true,$(INCLUDE_LATEST)),$(CONTAINER_REGISTRY)/$(IMAGE_NAME):latest)

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
