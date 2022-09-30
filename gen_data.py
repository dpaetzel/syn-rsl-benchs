import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import click  # type: ignore
import scipy.stats as st  # type: ignore


@click.command()
@click.option("-K",
              "--n-components",
              default=3,
              type=int,
              help="Number of components to use")
@click.option("-d",
              "--dimensions",
              default=1,
              type=int,
              help="Number of input dimensions to generate data for")
@click.option("-s",
              "--seed",
              default=1,
              type=int,
              help="Random seed to be used")
@click.option("--crowd-reg-radius",
              default=0.25,
              type=float,
              help="99% radius of the crowded region")
@click.argument("N", type=int)
def cli(n_components, dimensions, seed, n, crowd_reg_radius):

    np.random.seed(seed)

    # TODO Generate or configure the number of crowded regions first.

    # Scale was determined visually.
    center_crowded_region = st.norm(
        loc=0.0, scale=0.2).rvs(dimensions).reshape(dimensions)

    cov_crowded_region = np.eye(dimensions) * (
        crowd_reg_radius**2 / st.chi2(df=dimensions).ppf(q=0.99))

    # Generate one less rule because we add a default rule later.
    centers = st.multivariate_normal(
        mean=center_crowded_region,
        cov=cov_crowded_region).rvs(n_components - 1).reshape(
            n_components - 1, dimensions)

    # TODO How much overlap do we want?
    spreads = st.uniform().rvs((n_components - 1, dimensions))

    # Add a default rule so we don't have to check whether there is a rule
    # matching.
    centers = np.vstack([np.repeat(0, dimensions), centers])
    spreads = np.vstack([np.repeat(1, dimensions), spreads])

    def match(x):
        """
        Values of the `n_components` matching functions for the given input.
        """

        # One condition per rule (rows) per dimension (columns).
        conds = (centers - spreads <= x) & (x <= centers + spreads)

        # A rule only matches if all its conditions are fulfilled.
        return np.all(conds, axis=1).astype(float)

    # d coefficients per rule.
    coeffs = st.uniform(loc=-4, scale=8).rvs((n_components, dimensions))

    # One intercept per rule.
    intercepts = st.uniform(loc=-4, scale=8).rvs(n_components)

    def output_local(x):
        """
        Values of the `n_components` local models for the given input.
        """
        return np.sum(x * coeffs, axis=1) + intercepts

    # Noise is fixed per rule (also, assume same noise for each dimension).
    std_noises = st.gamma(a=1.0, scale=1.0).rvs(n_components)

    # One mixing coefficient per rule.
    mixing_weights = st.uniform().rvs(n_components)

    def output(x):
        """
        Output of the overall model for the given input (including local noise).
        """
        m = match(x)
        f = output_local(x)
        mixing = (mixing_weights * m) / np.sum(mixing_weights * m)
        noise = st.norm(loc=0.0, scale=std_noises).rvs()
        return np.sum(mixing * (f + noise))

    # TODO Print how much overlap we have.

    X = st.uniform(loc=-1, scale=2).rvs((n, dimensions))

    y = []
    for x in X:
        y.append(output(x))

    X = pd.DataFrame(X).rename(columns=lambda i: f"X{i}")
    y = pd.Series(y).rename("y")
    print(pd.concat([X, y], axis=1).to_csv(index=False))

    if dimensions == 1:
        import matplotlib.pyplot as plt  # type: ignore
        X = np.linspace(-1, 1, 1000)
        y = []
        for x in X:
            y.append(output(x))

        fig, ax = plt.subplots()
        ax.scatter(X, y, label="data")
        ax.plot(X,
                st.norm(loc=center_crowded_region,
                        scale=np.sqrt(cov_crowded_region)[0]).pdf(X),
                color="C1",
                linestyle="dotted",
                label="crowdedness")
        ax.vlines(centers.ravel(),
                  ymin=min(y),
                  ymax=max(y),
                  color="C2",
                  linestyle="dashed",
                  label="component centers")
        ax.set_xlabel("inputs (X)")
        ax.set_ylabel("outputs (y)")
        ax.legend()

        plt.show()


if __name__ == "__main__":
    cli()
