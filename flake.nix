{
  description = "bazarr-whisper-proxy — Bazarr-to-OpenArc ASR bridge";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = nixpkgs.lib;
        python = pkgs.python314;

        # ---------- uv2nix workspace & base overlay ----------

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # ---------- CPU-only overlay: make CUDA/GPU packages no-ops ----------
        # torch and its deps reference these on Linux; they're not needed for
        # CPU-only inference and would add several GB to the image.
        cpuOnlyOverlay =
          final: prev:
          let
            mkNoop =
              name:
              pkgs.runCommandLocal "${name}-noop" {
                passthru = {
                  dependencies = { };
                  optional-dependencies = { };
                  dependency-groups = { };
                };
              } "mkdir $out";

            cudaPackages = [
              "cuda-bindings"
              "cuda-pathfinder"
              "nvidia-cublas"
              "nvidia-cuda-cupti"
              "nvidia-cuda-nvrtc"
              "nvidia-cuda-runtime"
              "nvidia-cudnn-cu13"
              "nvidia-cufft"
              "nvidia-cufile"
              "nvidia-curand"
              "nvidia-cusolver"
              "nvidia-cusparse"
              "nvidia-cusparselt-cu13"
              "nvidia-nccl-cu13"
              "nvidia-nvjitlink"
              "nvidia-nvshmem-cu13"
              "nvidia-nvtx"
              "triton"
            ];
          in
          # cuda-toolkit has extras that torch[linux] depends on;
          # make each extra resolve to an empty dep set so the resolver
          # doesn't error on "extra not found".
          {
            "cuda-toolkit" = pkgs.runCommandLocal "cuda-toolkit-noop" {
              passthru = {
                dependencies = { };
                optional-dependencies = lib.genAttrs [
                  "cudart"
                  "cufft"
                  "cufile"
                  "cupti"
                  "curand"
                  "cusolver"
                  "cusparse"
                  "nvjitlink"
                  "nvrtc"
                  "nvtx"
                ] (_: { });
                dependency-groups = { };
              };
            } "mkdir $out";
          }
          // lib.genAttrs cudaPackages mkNoop;

        # ---------- whisper-proxy stub: deps graph but no source code ----------
        # This keeps the Python-deps venv stable across app-source changes.
        # The stub preserves passthru.dependencies so transitive deps
        # (fastapi, torch, …) are still included in the virtualenv.
        whisperProxyStub =
          final: prev:
          let
            orig = prev."whisper-proxy";
          in
          {
            "whisper-proxy" = pkgs.runCommandLocal "whisper-proxy-stub" {
              passthru = {
                dependencies = orig.passthru.dependencies or { };
                optional-dependencies = orig.passthru.optional-dependencies or { };
                dependency-groups = orig.passthru.dependency-groups or { };
              };
            } "mkdir $out";
          };

        # ---------- Python package sets ----------

        # ctc-forced-aligner ships only an sdist whose pyproject.toml omits
        # setuptools from [build-system], so inject it here.
        # numba's wheel links tbbpool.so against libtbb.so.12 (oneTBB);
        # add pkgs.tbb so auto-patchelf can patch the RPATH correctly.
        buildSystemFixesOverlay =
          final: prev: {
            # These sdist-only packages use setuptools as their build backend
            # but don't declare it in [build-system]; inject it via
            # nativeBuildInputs so its setup-hook populates NIX_PYPROJECT_PYTHONPATH.
            "ctc-forced-aligner" = prev."ctc-forced-aligner".overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
            });
            "pysrt" = prev."pysrt".overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
            });
            # numba's wheel links tbbpool.so against libtbb.so.12 (oneTBB);
            # add pkgs.tbb so auto-patchelf can patch the RPATH correctly.
            "numba" = prev."numba".overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb ];
            });
            # torch ships a unified CPU+CUDA wheel; the CUDA .so files link
            # against libcudart, libcublas, etc. which are not present in a
            # CPU-only image.  Ignore those missing CUDA deps at patchelf time;
            # the CUDA code paths will never be called.
            "torch" = prev."torch".overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;
            });
          };

        basePythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
          cpuOnlyOverlay
          buildSystemFixesOverlay
        ]);

        # Full venv (real app source; used for packages.default)
        appVenv = basePythonSet.mkVirtualEnv "whisper-proxy-env" workspace.deps.default;

        # Deps-only venv for the image (stub replaces app source; stable layer)
        imagePythonSet = basePythonSet.overrideScope whisperProxyStub;
        depsVenv = imagePythonSet.mkVirtualEnv "whisper-proxy-deps" workspace.deps.default;

        # ---------- Aligner model ----------

        # MMS CTC forced-aligner ONNX weights — baked as a dedicated image layer
        # so app rebuilds don't invalidate this large (1.2 GB) cached layer.
        alignerModel = pkgs.fetchurl {
          name = "ctc-forced-aligner-model.onnx";
          url = "https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx";
          hash = "sha256-6LrWf9NTOz08FFsMoxuxU4OUXBM4TdiXW6qntz97Yaw=";
        };

        # Place the model at /models/model.onnx inside its own store path so it
        # becomes a dedicated Docker layer independent of the app and Python deps.
        alignerModelDir = pkgs.runCommandLocal "aligner-model-dir" { } ''
          mkdir -p $out/models
          cp ${alignerModel} $out/models/model.onnx
        '';

        # ---------- App source layer ----------

        # Only the Python source tree — the derivation hash changes with source
        # changes, giving it its own Docker layer separate from depsVenv.
        appSrc = pkgs.runCommandLocal "whisper-proxy-src" { } ''
          mkdir -p $out/src
          cp -r ${./src}/whisper_proxy $out/src/whisper_proxy
        '';

        # ---------- OCI image ----------

        dockerImage = pkgs.dockerTools.streamLayeredImage {
          name = "bazarr-whisper-proxy";
          tag = "latest";

          # Layer order (bottom → top):
          #   system (tini, ca-certs) — rarely changes
          #   depsVenv               — changes when PyPI deps change
          #   alignerModelDir        — changes when model changes (rare)
          #   appSrc                 — changes every commit
          contents = [
            pkgs.tini
            pkgs.cacert
            depsVenv
            alignerModelDir
            appSrc
          ];

          config = {
            # tini as PID 1: forwards signals and reaps zombies.
            Entrypoint = [
              "${pkgs.tini}/bin/tini"
              "--"
            ];
            Cmd = [
              "${depsVenv}/bin/python"
              "-m"
              "whisper_proxy"
            ];
            Env = [
              # Real app source shadows any stub in the venv's site-packages.
              "PYTHONPATH=${appSrc}/src"
              # Pre-baked model; no network fetch at startup.
              "ALIGNER_MODEL_PATH=${alignerModelDir}/models/model.onnx"
              # TLS roots for HTTPS calls to OpenArc.
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            ];
            ExposedPorts = {
              "9000/tcp" = { };
            };
          };
        };

        # ---------- ctc-forced-aligner from PyPI (not in nixpkgs) ----------
        # Needed by tests that import from whisper_proxy.aligner; build it
        # from the same sdist hash pinned in uv.lock.
        ctcForcedAligner = python.pkgs.buildPythonPackage {
          pname = "ctc-forced-aligner";
          version = "1.0.2";
          pyproject = true;
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/5f/5a/0cf21de3ddc9f2696039063a290f8a4bb9059c9ac64fa325b08ef2769efd/ctc_forced_aligner-1.0.2.tar.gz";
            hash = "sha256-i7hjMWrU7jCijwAiezjJuEUd9TCXqCpw0p9100/2t/8=";
          };
          build-system = [ python.pkgs.setuptools ];
          propagatedBuildInputs = with python.pkgs; [
            torch
            numpy
            onnxruntime
            librosa
          ];
          doCheck = false;
        };

        # ---------- Dev env (unchanged from original) ----------

        checkPythonEnv = python.withPackages (ps: with ps; [
          # runtime deps
          fastapi
          uvicorn
          httpx
          pydantic-settings
          numpy
          soundfile
          pysrt
          pycountry
          unidecode
          python-json-logger
          python-multipart
          onnxruntime
          torch
          # test deps
          mypy
          pytest
          pytest-asyncio
          anyio
          respx
          pysubs2
        ] ++ [ ctcForcedAligner ]);

      in
      {
        packages = {
          default = appVenv;
          inherit dockerImage;
          aligner-model = alignerModel;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.ruff
            checkPythonEnv
            pkgs.ffmpeg
            pkgs.docker
            pkgs.docker-compose
            pkgs.gh
          ];

          shellHook = ''
            export UV_PYTHON="${python}/bin/python3"

            DOCKER_SOCK="/tmp/docker.sock"
            export DOCKER_HOST="unix://$DOCKER_SOCK"
            if [ ! -S "$DOCKER_SOCK" ]; then
              echo "Starting dockerd on $DOCKER_SOCK..."
              sudo sh -c "dockerd --data-root /tmp/docker-data --host unix://$DOCKER_SOCK --storage-driver vfs &>/tmp/dockerd.log &"
              for i in $(seq 1 10); do
                [ -S "$DOCKER_SOCK" ] && break
                sleep 1
              done
            fi
            if [ -S "$DOCKER_SOCK" ]; then
              sudo chmod 666 "$DOCKER_SOCK"
            fi
          '';
        };

        checks = {
          lint = pkgs.runCommand "whisper-proxy-lint"
            {
              src = self;
              nativeBuildInputs = [
                pkgs.ruff
                checkPythonEnv
              ];
            }
            ''
              export HOME=$(mktemp -d)
              export RUFF_CACHE_DIR="$HOME/.ruff_cache"
              export MYPY_CACHE_DIR="$HOME/.mypy_cache"
              cd "$src"
              ruff check src tests
              ruff format --check src tests
              mypy --strict --no-incremental src
              touch "$out"
            '';

          tests = pkgs.runCommand "whisper-proxy-tests"
            {
              src = self;
              nativeBuildInputs = [ checkPythonEnv ];
              PYTHONPATH = "${self}/src";
              # The Nix sandbox has no system CA store; point to nixpkgs bundle
              # so httpx (used by OpenArcClient) can init its SSL context.
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              # Skip tests that require the real ONNX model (network download).
              SKIP_ALIGNER_INTEGRATION = "1";
            }
            ''
              export HOME=$(mktemp -d)
              cd "$src"
              pytest tests/ -v
              touch "$out"
            '';
        };
      }
    );
}
