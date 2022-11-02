import sys

import click  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.stats as st  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import (
    mean_absolute_error,  # type: ignore
    mean_squared_error)


# https://stackoverflow.com/a/14981125/6936216
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
              default=1,
              type=float,
              help="99% radius of the crowded region")
@click.argument("N", type=int)
def cli(n_components, dimensions, seed, n, crowd_reg_radius):

    np.random.seed(seed)

    # Minimum volume of an inteval is the volume of a cuboid with an edge length
    # of one tenth the input space extent.
    volume_interval_min = ((1.0 - (-1.0)) / 10.0)**dimensions

    def overlap(int1, int2):
        """
        Compute the overlap between the two intervals.

        Parameters
        ----------
        int1, int2: array
            An array of shape (2, `dimensions`). I.e. `int1[0]` is the lower
            bound and `int1[1]` is the upper bound of the interval.

        Returns
        -------
        array or None
            If the intervals do not overlap, return `None`. Otherwise return the
            overlap interval.
        """
        l1 = int1[0]
        u1 = int1[1]
        l2 = int2[0]
        u2 = int2[1]

        l = np.max([l1, l2], axis=0)
        u = np.min([u1, u2], axis=0)

        if np.any(u < l):
            return None
        else:
            return np.vstack([l, u])

    def volume(interval):
        """
        Compute the volume covered by the given interval.
        """
        return np.prod(interval[1] - interval[0])

    def draw_interval():
        """
        Draw an interval with a volume of at least `volume_interval_min`.
        """
        bounds = st.uniform(np.repeat(-1.0, dimensions), 2).rvs(
            (2, dimensions))

        # Sort each dimension independently.
        interval = np.sort(bounds.T, axis=1).T

        if volume(interval) < volume_interval_min:
            draw_interval()

        return interval

    intervals = []
    overlaps = []
    volumes_overlaps = []

    while len(intervals) < n_components - 1:
        interval = draw_interval()
        new_overlaps = []
        for existing_interval in intervals:
            new_overlaps.append(overlap(interval, existing_interval))

        volume_overlap = np.sum(
            [volume(overlap) for overlap in new_overlaps if overlap is not None])
        # Only use the interval if it adds overlap volume of at most the volume
        # of a cube having one tenth of the input space.
        if volume_overlap < volume_interval_min:
            intervals.append(interval)
            overlaps += [o for o in new_overlaps if o is not None]
            volumes_overlaps.append(volume_overlap)
        else:
            eprint(
                f"Rejecting due to overlap of {volume_overlap} (currently {len(intervals)} intervals)"
            )

    intervals = np.reshape(intervals, (n_components - 1, 2, dimensions))

    centers = (intervals[:, 0, :] + intervals[:, 1, :]) / 2
    spreads = (intervals[:, 1, :] - intervals[:, 0, :]) / 2

    # Add a default rule so we don't have to check whether there is a rule
    # matching.
    centers = np.vstack([np.repeat(0, dimensions), centers])
    spreads = np.vstack([np.repeat(1, dimensions), spreads])
    eprint(f"Centers:\n{centers}\n")
    eprint(f"Spreads:\n{spreads}\n")

    eprint(f"Sum of overlaps:", sum(volumes_overlaps), "\n")

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
    std_noises = st.gamma(a=1.0, scale=0.1).rvs(n_components)

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
    y = [output(x) for x in X]

    X = pd.DataFrame(X).rename(columns=lambda i: f"X{i}")
    y = pd.Series(y).rename("y")
    print(pd.concat([X, y], axis=1).to_csv(index=False))

    model = LinearRegression()
    model.fit(X, y)

    eprint("Linear model:")
    eprint("coef_:", model.coef_)
    eprint("intercept_:", model.intercept_)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    eprint("MAE", mae)
    eprint("MSE", mse)
    eprint("\n")

    if dimensions == 2:
        import matplotlib.cm
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.collections import PatchCollection  # type: ignore
        from matplotlib.patches import Rectangle  # type: ignore

        fig, ax = plt.subplots()

        # We remove the default rule here.
        centers_ = centers[1:]
        spreads_ = spreads[1:]
        xys = centers_ - spreads_
        widths = 2 * spreads_[:, 0]
        heights = 2 * spreads_[:, 1]

        boxes = [
            Rectangle(xy, width, height)
            for xy, width, height in zip(xys, widths, heights)
        ]
        pc = PatchCollection(boxes, cmap=matplotlib.cm.jet, alpha=0.8)
        pc.set_array(100 * np.random.random(n_components - 1))
        ax.add_collection(pc)
        ax.set_xbound(lower=-1, upper=1)
        ax.set_ybound(lower=-1, upper=1)

        plt.show()

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

        ax.hlines(
            np.linspace(
                min(y) - 0.1,
                min(y) - 0.1 - n_components * 0.1, n_components),
            centers - spreads, centers + spreads)

        # TODO Paint rules directly into scatter plot

        ax.set_xlabel("inputs (X)")
        ax.set_ylabel("outputs (y)")
        ax.legend()

        plt.show()


if __name__ == "__main__":
    cli()
