import sys

import click  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.stats as st  # type: ignore
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
              "--dimension",
              default=1,
              type=int,
              help="Input dimension to generate data for")
@click.option("-s",
              "--seed",
              default=1,
              type=int,
              help="Random seed to be used")
@click.option("--restrict-overlap/--no-restrict-overlap",
              default=False,
              help="Whether to restrict overlap of the components")
@click.argument("N", type=int)
def cli(n_components, dimension, seed, n, restrict_overlap):
    if restrict_overlap:
        raise NotImplementedError("Restricting overlap is not properly "
                                  "calibrated to dimension right now.")

    np.random.seed(seed)

    volume_input_space = (1.0 - (-1.0))**dimension

    # Minimum interval volume is chosen to be a percentage of the volume per
    # component.
    factor = 1.0 / (n_components - 1) / 10.0
    volume_interval_min = factor * volume_input_space

    def overlap(int1, int2):
        """
        Compute the overlap between the two intervals.

        Parameters
        ----------
        int1, int2: array
            An array of shape (2, `dimension`). I.e. `int1[0]` is the lower
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
        # TODO Make spreads depend on the other already drawn intervals so we
        # don't have to reject as many (i.e. if one dimension is already pretty
        # full, consider to make drawn intervals small in that dimension)

        # TODO Consider making spread_min depend on dimension
        spread_min = 0.1
        # The hard maximum for interval spread in each dimension is 1.
        #
        # Beta distribution's a slightly larger then b in order to bias towards
        # >0.5 (with a=12 and b=10, only 17%ish of probability mass is below
        # 0.5) which leads to higher probability of spreads >1 which reduces
        # pressure on the last interval (and thus less rejections).
        # dist_spread = st.beta(12, 10, loc=spread_min, scale=1.0 - spread_min)
        #
        # Since we don't draw a fixed volume (and then compute a fixed width of
        # the last dimension interval) but instead only compute a minimum width
        # for the last dimension interval we may as well use a simple uniform
        # distribution here.
        dist_spread = st.uniform(spread_min, scale=1.0 - spread_min)
        spreads = dist_spread.rvs(dimension - 1)

        centers = st.uniform(-1 + spreads, 2 - 2 * spreads).rvs(dimension - 1)

        interval = np.array([centers - spreads, centers + spreads])

        # Compute the minimum width of the last interval.
        min_width = volume_interval_min / volume(interval)

        max_width = 1. - (-1.)

        # While the minimum width computed is larger than the maximum width,
        # redraw the smallest already chosen spread.
        iter_max = 20
        i = 0
        while min_width > max_width and i < iter_max:
            i += 1
            eprint("Rejecting due to min width greater max width "
                   f"({min_width} > {max_width}).")
            i = np.argmin(spreads)
            new_spread = dist_spread.rvs()
            print(i, spreads[i], new_spread, spreads)
            spreads[i] = new_spread
            centers[i] = st.uniform(-1 + spreads[i], 2 - 2 * spreads[i]).rvs()

            interval = np.array([centers - spreads, centers + spreads])
            min_width = volume_interval_min / volume(interval)

        if i >= iter_max:
            eprint("Had to reject too many, aborting.")
            sys.exit(1)

        # Finally, we may draw a random width for the last interval.
        width = st.uniform(min_width, scale=max_width - min_width).rvs()

        # Compute the spread of the last interval.
        spread = width / 2

        # Draw the center for the last interval. In doing so, consider the
        # interval's spread and don't go too close to the edge of the input
        # space.
        center = st.uniform(-1 + spread, 2 - 2 * spread).rvs()

        # Append last dimension to the interval.
        interval_last = [center - spread, center + spread]
        interval = np.hstack(
            [interval, np.array(interval_last)[:, np.newaxis]])

        return interval

    intervals = []
    overlaps = []
    volumes_overlaps = []

    iter_max = 20
    i = 0
    while len(intervals) < n_components - 1 and i < iter_max:
        i += 1
        interval = draw_interval()
        new_overlaps = []
        for existing_interval in intervals:
            new_overlaps.append(overlap(interval, existing_interval))

        volume_overlap = np.sum([
            volume(overlap) for overlap in new_overlaps if overlap is not None
        ])
        # Only use the interval if it adds overlap volume of at most the volume
        # of a cube having one tenth of the input space.
        if not restrict_overlap or volume_overlap <= volume_interval_min:
            intervals.append(interval)
            overlaps += [o for o in new_overlaps if o is not None]
            volumes_overlaps.append(volume_overlap)
            i = 0
        else:
            eprint("Rejecting: Too much overlap with already chosen intervals "
                   f"({volume_overlap:.2f} > {volume_interval_min:.2f}, "
                   f"chose {len(intervals)} of "
                   f"{n_components - 1} intervals so far).")

    if i >= iter_max:
        eprint("Had to reject too many, aborting.")
        sys.exit(1)

    intervals = np.reshape(intervals, (n_components - 1, 2, dimension))

    centers = (intervals[:, 0, :] + intervals[:, 1, :]) / 2
    spreads = (intervals[:, 1, :] - intervals[:, 0, :]) / 2

    # Add a default rule so we don't have to check whether there is a rule
    # matching.
    centers = np.vstack([np.repeat(0, dimension), centers])
    spreads = np.vstack([np.repeat(1, dimension), spreads])
    eprint(f"Centers:\n{centers}\n")
    eprint(f"Spreads:\n{spreads}\n")
    eprint(f"Volumes:\n{[volume(i) for i in intervals]}\n")

    eprint(f"Minimum interval volume: {volume_interval_min}\n")
    eprint(f"Sum of overlaps: {sum(volumes_overlaps)}\n")

    def match(x):
        """
        Values of the `n_components` matching functions for the given input.
        """

        # One condition per rule (rows) per dimension (columns).
        conds = (centers - spreads <= x) & (x <= centers + spreads)

        # A rule only matches if all its conditions are fulfilled.
        return np.all(conds, axis=1).astype(float)

    # d coefficients per rule.
    coeffs = st.uniform(loc=-4, scale=8).rvs((n_components, dimension))

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

    X = st.uniform(loc=-1, scale=2).rvs((n, dimension))
    y = [output(x) for x in X]

    counts_match = np.sum([match(x) for x in X], axis=0)

    eprint(f"Match counts: {counts_match}\n")

    matchs = [match(x) for x in st.uniform(loc=-1, scale=2).rvs(500_000)]
    # Drop the default rule entries.
    matchs = np.array(matchs)[:, 1:]
    # Count how many rules match each input.
    matchs = np.sum(matchs, axis=1)
    ratio_vol_covered = np.sum(matchs != 0) / len(matchs)

    eprint(
        f"Percentage of volume covered (MC approximation): {ratio_vol_covered * 100:.1f} %\n"
    )

    X = pd.DataFrame(X).rename(columns=lambda i: f"X{i}")
    y = pd.Series(y).rename("y")
    print(pd.concat([X, y], axis=1).to_csv(index=False))

    model = make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(regressor=LinearRegression(),
                                   transformer=StandardScaler()))
    model.fit(X, y)

    eprint("Linear model:")
    eprint("coef_ (in standardized space):", model[1].regressor_.coef_)
    eprint("intercept_ (in standardized space):",
           model[1].regressor_.intercept_)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    eprint(f"MAE (on training data): {mae:.2f}")
    eprint(f"MSE (on training data): {mse:.2f}")
    eprint(f"R^2 (on training data): {r2:.2f}")
    eprint("\n")

    if dimension == 2:
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

    if dimension == 1:
        import matplotlib.pyplot as plt  # type: ignore
        X = np.linspace(-1, 1, 1000)
        y = []
        for x in X:
            y.append(output(x))

        fig, ax = plt.subplots()
        ax.scatter(X, y, label="data")
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
