{
  description = "bazarr-whisper-proxy — Bazarr-to-OpenArc ASR bridge";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;

        # Minimal Python env for checks — only what lint + tests need.
        # Heavy runtime deps (torch, ctc-forced-aligner) are managed by uv.
        checkPythonEnv = python.withPackages (ps: with ps; [
          fastapi
          uvicorn
          httpx
          mypy
          pytest
          pytest-asyncio
          anyio
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.ruff
            checkPythonEnv
            pkgs.ffmpeg
          ];

          shellHook = ''
            export UV_PYTHON="${python}/bin/python3"
          '';
        };

        checks = {
          lint = pkgs.runCommand "whisper-proxy-lint"
            {
              src = self;
              nativeBuildInputs = [ pkgs.ruff checkPythonEnv ];
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
            }
            ''
              export HOME=$(mktemp -d)
              cd "$src"
              pytest tests/ -v
              touch "$out"
            '';
        };
      });
}
