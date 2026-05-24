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

        # Only include files uv2nix needs to resolve the workspace.  This
        # prevents changes to tests/, docs/, scripts/, etc. from producing a
        # new store path for workspaceRoot, which would otherwise invalidate
        # the Python-deps layer in the OCI image even when no package
        # dependencies have changed.
        filteredWorkspaceRoot = lib.cleanSourceWith {
          src = ./.;
          filter =
            path: _type:
            let
              rel = lib.removePrefix (toString ./. + "/") (toString path);
            in
            rel == "pyproject.toml" || rel == "uv.lock" || lib.hasPrefix "src/" rel;
        };

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = filteredWorkspaceRoot; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

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
          final: prev:
          let
            mkNoop = name: pkgs.runCommandLocal "${name}-noop" {
              passthru = {
                dependencies = { };
                optional-dependencies = { };
                dependency-groups = { };
              };
            } "mkdir $out";
          in
          {
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
            # torch declares sympy (symbolic math) and networkx (graph ops) as
            # runtime deps, but both are only used by torch.compile() / torch.fx
            # which are never called during forced-alignment inference.
            "sympy" = mkNoop "sympy";
            "networkx" = mkNoop "networkx";
            # openvino's wheel ships libopenvino_intel_gpu_plugin.so linked
            # against libOpenCL.so.1; that lib is provided at runtime by
            # ocl-icd via LD_LIBRARY_PATH (set in the OCI image config), so
            # tell auto-patchelf to ignore it at build time.
            "openvino" = prev."openvino".overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [
                "libOpenCL.so.1"
              ];
            });
            # torch ships test executables, C++ extension headers, and test .so
            # files inside the wheel — none are needed for inference at runtime.
            "torch" = prev."torch".overrideAttrs (old: {
              postInstall = (old.postInstall or "") + ''
                site=$out/lib/python*/site-packages/torch
                rm -rf $site/test $site/include $site/bin
                rm -f  $site/lib/libtorchbind_test.so $site/lib/libjitbackend_test.so
              '';
            });
          };

        basePythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
          buildSystemFixesOverlay
        ]);

        # Full venv (real app source; used for packages.default)
        appVenv = basePythonSet.mkVirtualEnv "whisper-proxy-env" workspace.deps.default;

        # Deps-only venv for the image (stub replaces app source; stable layer)
        imagePythonSet = basePythonSet.overrideScope whisperProxyStub;
        depsVenv = imagePythonSet.mkVirtualEnv "whisper-proxy-deps" workspace.deps.default;

        # ---------- Aligner model ----------

        # MMS CTC forced-aligner ONNX weights — converted to OpenVINO IR at
        # build time so the runtime doesn't pay the read_model/serialize cost
        # on container start.
        alignerOnnxModel = pkgs.fetchurl {
          name = "ctc-forced-aligner-model.onnx";
          url = "https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx";
          hash = "sha256-6LrWf9NTOz08FFsMoxuxU4OUXBM4TdiXW6qntz97Yaw=";
        };

        # Convert ONNX → OpenVINO IR (model.xml + model.bin). Uses nixpkgs'
        # openvino for the build step; the runtime uses the PyPI wheel via
        # uv2nix. Both produce/consume the same IR format.
        alignerIRConvertPython = python.withPackages (ps: [ ps.openvino ]);
        alignerIRConvertScript = pkgs.writeText "convert-onnx-to-ir.py" ''
          import os, sys, openvino as ov
          onnx_path, xml_path, bin_path = sys.argv[1:4]
          model = ov.Core().read_model(onnx_path)
          ov.serialize(model, xml_path, bin_path)
        '';
        alignerIRDir = pkgs.runCommand "aligner-ir-dir" { } ''
          mkdir -p $out/models
          ${alignerIRConvertPython}/bin/python ${alignerIRConvertScript} \
            ${alignerOnnxModel} \
            $out/models/model.xml \
            $out/models/model.bin
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
          #   system (tini, ca-certs, GPU userspace) — rarely changes
          #   depsVenv                              — changes when PyPI deps change
          #   alignerIRDir                          — changes when model changes (rare)
          #   appSrc                                — changes every commit
          contents = [
            pkgs.tini
            pkgs.cacert
            # OpenVINO GPU plugin dlopens libOpenCL.so.1 (from ocl-icd) which
            # in turn loads the Intel Neo driver (libigdrcl.so) referenced by
            # the ICD manifest under intel-compute-runtime/etc/OpenCL/vendors.
            pkgs.ocl-icd
            pkgs.intel-compute-runtime
            depsVenv
            alignerIRDir
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
              # Pre-baked OpenVINO IR; no conversion at startup.
              "ALIGNER_MODEL_PATH=${alignerIRDir}/models/model.xml"
              # Persists OpenVINO's compiled-blob cache across container
              # restarts so first request after restart doesn't recompile.
              # Optional.
              # "ALIGNER_CACHE_DIR=/var/cache/aligner"
              # TLS roots for HTTPS calls to OpenArc.
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              # The OpenVINO wheel ships libs that link against libstdc++ and
              # the GPU plugin dlopens libOpenCL.so.1. Both need to be on the
              # loader's search path at runtime.
              "LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.ocl-icd}/lib:${pkgs.intel-compute-runtime}/lib/intel-opencl"
              # OpenCL ICD loader looks here for vendor manifests (.icd files).
              "OCL_ICD_VENDORS=${pkgs.intel-compute-runtime}/etc/OpenCL/vendors"
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
            # Library imports onnxruntime at module load; kept as a dep even
            # though forced-alignment inference is now routed through OpenVINO.
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
          openvino
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
          aligner-onnx = alignerOnnxModel;
          aligner-ir = alignerIRDir;
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
            # e2e harness: synthesizes fixture media (tests/e2e/fixtures/build.py)
            pkgs.espeak-ng
            # e2e harness: convenient REST poking from scripts/e2e.sh
            pkgs.curl
            pkgs.jq
          ];

          shellHook = ''
            export UV_PYTHON="${python}/bin/python3"
            # Put scripts/ on PATH so `e2e` (harness entrypoint) is callable.
            export PATH="$PWD/scripts:$PATH"

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
