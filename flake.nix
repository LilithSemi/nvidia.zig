{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flakever.url = "github:numinit/flakever";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      flakever,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      nameValuePair = name: value: { inherit name value; };
      genAttrs = names: f: builtins.listToAttrs (map (n: nameValuePair n (f n)) names);
      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      flakeverConfig = flakever.lib.mkFlakever {
        inherit inputs;

        digits = [
          1
          2
          2
        ];
      };

      forAllSystems =
        f:
        genAttrs allSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs { inherit system; };
          }
        );

      treefmtEval = forAllSystems ({ pkgs, ... }: treefmt-nix.lib.evalModule pkgs (import ./treefmt.nix));
    in
    {
      versionTemplate = "1.1pre-<lastModifiedDate>-<rev>";

      devShells = forAllSystems (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            name = "nvidia-zig-dev-shell";
            packages = with pkgs; [
              zig_0_17
            ];
          };
        }
      );

      formatter = forAllSystems ({ system, ... }: treefmtEval.${system}.config.build.wrapper);

      checks = forAllSystems (
        { system, pkgs, ... }:
        {
          default = pkgs.stdenv.mkDerivation (finalAttrs: {
            pname = "nvidia-zig";
            inherit (flakeverConfig) version;

            src = lib.cleanSource ./.;

            zigDeps = pkgs.zig_0_17.fetchDeps {
              inherit (finalAttrs) src pname version;
              hash = "sha256-1qjGFk3Vq16sbX6u2ZEZWKdtweilugIk30uirT4olnM=";
            };

            nativeBuildInputs = with pkgs; [
              zig_0_17
            ];

            postConfigure = ''
              ln -s ${finalAttrs.zigDeps} "$ZIG_GLOBAL_CACHE_DIR/p"
            '';

            doCheck = true;
          });

          formatting = treefmtEval.${system}.config.build.check self;
        }
      );
    };
}
