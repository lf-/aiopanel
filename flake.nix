{
  description = "Panel written in asyncio";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # for mystery reasons, you can't update a nested flake input??
    pypi-deps-db = {
      owner = "DavHau";
      repo = "pypi-deps-db";
      type = "github";
    };
    mach-nix = {
      url = "github:DavHau/mach-nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
    };
    aiobspwm = {
      url = "github:lf-/aiobspwm";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, aiobspwm, ... }:
    let
      out = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            # overlays = [ self.overlays.default ];
          };

          requirements = ''
            pydbus
            gbulb
            jinja2
            pulsectl
          '';

          providers = {
            pulsectl = "nixpkgs";
            gbulb = "nixpkgs";
          };

          packagesExtra =
            [ self.packages.${system}.aiobspwm ];
        in
        {
          packages.default = mach-nix.lib."${system}".buildPythonApplication {
            pname = "aiopanel";
            src = ./.;
            inherit requirements providers packagesExtra;
            version = "0.0.1";
          };
          packages.aiobspwm = mach-nix.lib."${system}".buildPythonApplication {
            pname = "aiobspwm";
            src = aiobspwm;
            requirements = " ";
            version = "0.0.1";
          };

          inherit pkgs;

          # tools that should be added to the shell
          devShells.default = mach-nix.lib."${system}".mkPythonShell {
            requirements = ''
              ${requirements}

              ipython
            '';
            inherit providers packagesExtra;
          };
        };
    in
    flake-utils.lib.eachDefaultSystem out;
}
