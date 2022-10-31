{
  inputs = {
    # I typically use the exact nixpkgs set that I use for building my current
    # system to avoid redundancy.
    nixos-config.url = "github:dpaetzel/nixos-config";
  };

  outputs = { self, nixos-config }:
    let
      nixpkgs = nixos-config.inputs.nixpkgs;
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;
    in rec {
      devShell.${system} = pkgs.mkShell {

        buildInputs = with python.pkgs; [
          ipython
          python

          click
          matplotlib
          mlflow
          networkx
          numpy
          pandas
          scipy
          seaborn
        ];
      };
    };
}
