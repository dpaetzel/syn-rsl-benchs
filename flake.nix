{
  description = "Flake";

  inputs.nixpkgs.url =
    # 2022-06-22
    "github:NixOS/nixpkgs/0d68d7c857fe301d49cdcd56130e0beea4ecd5aa";

  outputs = { self, nixpkgs }:
    let system = "x86_64-linux";
    in with import nixpkgs {
      inherit system;
    };

    let python = python310;
    in rec {

      devShell.${system} = pkgs.mkShell {

        packages = with python.pkgs; [
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
