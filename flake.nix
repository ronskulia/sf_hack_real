# flake.nix
{
  description = "Pursuit-Evasion Phase Transition Study — multi-agent game theory with RL";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;  # PyTorch wheels sometimes flag as unfree
          };
        };

        # ---------- Python environment ----------
        python = pkgs.python312;

        # Core dependencies matching requirements.txt
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          torch          # CPU build from nixpkgs; sufficient for small MLPs
          matplotlib
          pyyaml
          tqdm
          imageio        # for animation frame assembly

          # Dev/QoL
          ipython
          pytest
        ]);

        # imageio-ffmpeg is fragile under Nix — we supply system ffmpeg instead.
        # Matplotlib's FuncAnimation can use ffmpeg directly when it's on PATH.
        runtimeDeps = with pkgs; [
          ffmpeg-full    # animation rendering (MP4 via FuncAnimation)
        ];

      in
      {
        # ── Development shell ──────────────────────────────────────────
        devShells.default = pkgs.mkShell {
          name = "pursuit-evasion";

          buildInputs = [ pythonEnv ] ++ runtimeDeps;

          shellHook = ''
            echo ""
            echo "  ╔══════════════════════════════════════════════════╗"
            echo "  ║  Pursuit-Evasion Phase Transition Study         ║"
            echo "  ║  Python ${python.version} + PyTorch (CPU) + ffmpeg      ║"
            echo "  ╚══════════════════════════════════════════════════╝"
            echo ""
            echo "  Quick start:"
            echo "    python scripts/quick_test.py          # smoke test (~30s)"
            echo "    python scripts/run_experiment.py \\    # full experiment"
            echo "           --config config.yaml"
            echo ""

            # Ensure Matplotlib uses the Agg backend (no display server needed)
            export MPLBACKEND=Agg

            # Point Matplotlib at system ffmpeg explicitly
            export FFMPEG_BINARY="${pkgs.ffmpeg-full}/bin/ffmpeg"

            # Keep __pycache__ out of the source tree if desired
            export PYTHONDONTWRITEBYTECODE=1

            # Add project root to PYTHONPATH so `from src import ...` works
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };

        # ── Runnable targets ───────────────────────────────────────────
        # `nix run .#quick-test` — smoke test without entering the shell
        packages.quick-test = pkgs.writeShellScriptBin "pursuit-evasion-quick-test" ''
          export MPLBACKEND=Agg
          export FFMPEG_BINARY="${pkgs.ffmpeg-full}/bin/ffmpeg"
          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONPATH="${self}:$PYTHONPATH"
          ${pythonEnv}/bin/python ${self}/scripts/quick_test.py "$@"
        '';

        # `nix run .#experiment` — full experiment
        packages.experiment = pkgs.writeShellScriptBin "pursuit-evasion-experiment" ''
          export MPLBACKEND=Agg
          export FFMPEG_BINARY="${pkgs.ffmpeg-full}/bin/ffmpeg"
          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONPATH="${self}:$PYTHONPATH"
          ${pythonEnv}/bin/python ${self}/scripts/run_experiment.py "$@"
        '';

        packages.default = self.packages.${system}.quick-test;

        # ── Checks (CI-friendly) ──────────────────────────────────────
        checks.default = pkgs.runCommand "pursuit-evasion-check" {
          buildInputs = [ pythonEnv ] ++ runtimeDeps;
        } ''
          export MPLBACKEND=Agg
          export PYTHONPATH="${self}:$PYTHONPATH"

          # Verify imports resolve
          ${pythonEnv}/bin/python -c "
          import numpy, torch, matplotlib, yaml, tqdm, imageio
          print('All imports OK')
          print(f'  NumPy:      {numpy.__version__}')
          print(f'  PyTorch:    {torch.__version__}')
          print(f'  Matplotlib: {matplotlib.__version__}')
          "

          # Verify ffmpeg is reachable
          ${pkgs.ffmpeg-full}/bin/ffmpeg -version | head -1

          mkdir -p $out
          echo "checks passed" > $out/result
        '';
      }
    );
}
