import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import click  # type: ignore
import scipy.stats as st  # type: ignore


@click.command()
@click.option("-K", "--n-rules", default=3, type=int)
@click.option("-d", "--dimensions", default=1, type=int)
@click.option("-s", "--seed", default=1, type=int)
@click.argument("N", type=int)
def cli(n_rules, dimensions, seed, n):

    np.random.seed(seed)

    # TODO Generate or configure the number of crowded regions first.
    # n_crowded_regions = ...
    # centers_crowded_regions = ...
    # centers = ...
    # spreads = ...

    # TODO Consider clipping to 0.99 or so (however, this has to depend on
    # dimensionality since in higher dimensions most of the space is further out
    # than 0.99)

    # TODO Several things should depend on dimensionality for [-1, 1] to make
    # sense as a domain

    # Scale was determined visually.
    center_crowded_region = st.norm(
        loc=0.0, scale=0.2).rvs(dimensions).reshape(dimensions)

    # Generate one less rule because we add a default rule later.
    centers = st.multivariate_normal(mean=center_crowded_region,
                                     cov=1).rvs(n_rules - 1).reshape(
                                         n_rules - 1, dimensions)

    # TODO How much overlap do we want?
    spreads = st.uniform().rvs((n_rules - 1, dimensions))

    # Add a default rule so we don't have to check whether there is a rule
    # matching.
    centers = np.vstack([np.repeat(0, dimensions), centers])
    spreads = np.vstack([np.repeat(1, dimensions), spreads])

    def match(x):
        """
        Values of the `n_rules` matching functions for the given input.
        """

        # One condition per rule (rows) per dimension (columns).
        conds = (centers - spreads < x) & (x < centers + spreads)

        # A rule only matches if all its conditions are fulfilled.
        return np.all(conds, axis=1).astype(float)

    # d coefficients per rule.
    coeffs = st.uniform(loc=-4, scale=8).rvs((n_rules, dimensions))

    # One intercept per rule.
    intercepts = st.uniform(loc=-4, scale=8).rvs(n_rules)

    def output_local(x):
        """
        Values of the `n_rules` local models for the given input.
        """
        return np.sum(x * coeffs, axis=1) + intercepts

    # Noise is fixed per rule.
    std_noises = st.norm(loc=0.0, scale=1.0).rvs(n_rules)

    # One mixing coefficient per rule.
    mixing_weights = st.uniform().rvs(n_rules)

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

if __name__ == "__main__":
    cli()
