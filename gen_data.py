import itertools
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
from sklearn.utils import check_random_state


# https://stackoverflow.com/a/14981125/6936216
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def intersection(int1, int2):
    """
    Computes the intersection between two intervals.

    Parameters
    ----------
    int1, int2: array
        An array of shape (2, `dimension`). I.e. `int1[0]` is the lower
        bound and `int1[1]` is the upper bound of the interval.

    Returns
    -------
    array or None
        If the intervals do not overlap, return `None`. Otherwise return the
        intersection interval.
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
    Computes the volume of the given interval.
    """
    return np.prod(interval[1] - interval[0])


def draw_interval(dimension, spread_min, volume_min, random_state):
    """
    Draws a random interval with a volume of at least `volume_interval_min`.

    Parameters
    ----------
    dimension : int > 0
        Dimension of the interval to be drawn.
    spread_min : float
        Minimum spread of the interval in all dimensions (the same for all
        dimensions).
    volume_min : float
        Minimum volume of the interval to be drawn.
    """
    # The hard maximum for interval spread in each dimension is 1. The upper
    # bound of `st.uniform` is `loc + scale` which is why we do `1.0 -
    # spread_min`.
    #
    # Since we don't draw a fixed volume (and then compute a fixed width of
    # the last dimension interval) but instead only compute a minimum width
    # for the last dimension interval we may as well use a simple uniform
    # distribution here.
    dist_spread = st.uniform(spread_min, scale=1.0 - spread_min)
    spreads = dist_spread.rvs(dimension - 1, random_state=random_state)

    centers = st.uniform(-1 + spreads,
                         2 - 2 * spreads).rvs(dimension - 1,
                                              random_state=random_state)

    interval = np.array([centers - spreads, centers + spreads])

    # Compute the minimum width of the interval in the last dimension.
    width_min = volume_min / volume(interval)

    # TODO Move this to top level constant.
    width_max = 1. - (-1.)

    # While the minimum width computed for the last dimension is larger than the
    # maximum width, redraw the smallest already chosen spread.
    iter_max = 20
    i = 0
    while width_min > width_max and i < iter_max:
        i += 1
        eprint("Rejecting due to min width greater max width "
               f"({width_min} > {width_max}).")
        i = np.argmin(spreads)
        new_spread = dist_spread.rvs(random_state=random_state)
        # eprint(i, spreads[i], new_spread, spreads)
        spreads[i] = new_spread
        centers[i] = st.uniform(
            -1 + spreads[i], 2 - 2 * spreads[i]).rvs(random_state=random_state)

        interval = np.array([centers - spreads, centers + spreads])
        width_min = volume_min / volume(interval)

    if i >= iter_max:
        eprint("Had to reject too many, aborting.")
        sys.exit(1)

    # Finally, we may draw a random width for the last dimension.
    width = st.uniform(width_min, scale=width_max
                       - width_min).rvs(random_state=random_state)

    # Compute the spread of the last dimension.
    spread = width / 2

    # Draw the center for the last dimension. In doing so, consider the
    # spread in that dimension and don't go too close to the edge of the input
    # space.
    center = st.uniform(-1 + spread,
                        2 - 2 * spread).rvs(random_state=random_state)

    # Append last dimension to the interval.
    interval_last = [center - spread, center + spread]
    interval = np.hstack([interval, np.array(interval_last)[:, np.newaxis]])

    return interval


def draw_intervals(dimension, n_intervals, volume_min, random_state):
    """
    Parameters
    ----------
    dimension : int > 0
    n_intervals : int > 0
    volume_min : float > 0
    random_state : np.random.RandomState

    Returns
    -------
    array, list, list
        The intervals as an array of shape `(n_intervals, 2, dimension)`, the
        set of pair-wise intersections between the intervals, the set of volumes
        of the non-empty ones of these pair-wise intersections.
    """
    intervals = []

    volume_avg = (1 - (-1))**dimension / n_intervals
    # If they were all cubes this is the spread in each dimension.
    spread_ideal_cubes = volume_avg**(1.0 / dimension) / 2.0
    spread_min = spread_ideal_cubes

    iter_max = 20
    i = 0
    while len(intervals) < n_intervals and i < iter_max:
        i += 1
        # TODO Consider makeing spreads depend on the other already drawn
        # intervals so we don't have to reject as many (i.e. if one dimension is
        # already pretty full, consider to make drawn intervals small in that
        # dimension)
        interval = draw_interval(dimension=dimension,
                                 spread_min=spread_min,
                                 volume_min=volume_min,
                                 random_state=random_state)
        # Only use the interval if it adds overlap volume of at most the volume
        # of a cube having one tenth of the input space.
        intervals.append(interval)

    intervals = np.reshape(intervals, (n_intervals, 2, dimension))

    return intervals


def overlap_volume(intervals):
    """
    The sum of the volumes of all the pairwise intersections of the given set of
    intervals.
    """
    pairs = itertools.combinations(intervals, 2)
    intersections = itertools.starmap(intersection, pairs)
    intersections_nonempty = [i for i in intersections if i is not None]
    return sum(map(volume, intersections_nonempty))


def centers_spreads(intervals):
    """
    Parameters
    ----------
    intervals : array of shape (n_intervals, 2, dimension)
        A set of intervals as a numpy array.

    Returns
    -------

    array, array
        Centers of the intervals, spreads of the intervals.
    """
    centers = (intervals[:, 0, :] + intervals[:, 1, :]) / 2.
    spreads = (intervals[:, 1, :] - intervals[:, 0, :]) / 2.

    return centers, spreads


def match(centers, spreads, x):
    """
    Values of the `n_components` matching functions for the given input.

    Parameters
    ----------
    centers : array
    spreads : array
    x : array
    """

    # One condition per rule (rows) per dimension (columns).
    conds = (centers - spreads <= x) & (x <= centers + spreads)

    # A rule only matches if all its conditions are fulfilled.
    return np.all(conds, axis=1).astype(float)


def outputs_local(coefs, intercepts, x):
    """
    Values of the `n_components` local models for the given input.
    """
    return np.sum(x * coefs, axis=1) + intercepts


def output_local(coefs, intercept, x):
    """
    Value of the local models defined by the given coefficients and intercept
    for the given input.
    """
    # Note that we could also re-use `outputs_local` here by wrapping stuff into
    # another array.
    return x @ coefs + intercept


def output(centers, spreads, coefs, intercepts, mixing_weights, std_noises,
           random_state, x):
    """
    Output of the overall model for the given input (including local noise).
    """
    m = match(centers=centers, spreads=spreads, x=x)

    # Probability is 0 for all rules that do not match due to multiplication
    # with `m`. Also, since we have a default rule, there will always be a
    # matching rule and we won't divide by zero here.
    p_responsible = (mixing_weights * m) / np.sum(mixing_weights * m)
    idx = random_state.choice(len(centers), p=p_responsible)

    y = output_local(coefs=coefs[idx], intercept=intercepts[idx], x=x)
    noise = st.norm(loc=0.0,
                    scale=std_noises[idx]).rvs(random_state=random_state)
    return y + noise


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
@click.argument("N", type=int)
@click.argument("NPZ", type=click.Path(dir_okay=False, writable=True))
def cli(n_components, dimension, seed, n, npz):
    """
    Build a model consisting of a set of components, use it to generate N data
    points and store the model parameters as well as the data points in the .npz
    file NPZ.
    """

    random_state = check_random_state(seed)

    volume_input_space = (1.0 - (-1.0))**dimension

    # Minimum interval volume is chosen to be a percentage of the volume per
    # component.
    factor = 1.0 / (n_components - 1) / 10.0
    volume_interval_min = factor * volume_input_space

    intervals = draw_intervals(
        dimension=dimension,
        # One less so we can add a default rule later on.
        n_intervals=n_components - 1,
        volume_min=volume_interval_min,
        random_state=random_state)

    centers, spreads = centers_spreads(intervals)

    volume_overlap = overlap_volume(intervals)

    # Add a default rule so we don't have to check whether there is a rule
    # matching.
    #
    # Important: Note that we do not add the default rule to `intervals`,
    # `overlaps`, `volumes_overlaps`.
    centers = np.vstack([np.repeat(0, dimension), centers])
    spreads = np.vstack([np.repeat(1, dimension), spreads])

    eprint(f"Centers:\n{centers}\n")
    eprint(f"Spreads:\n{spreads}\n")
    eprint(f"Volumes:\n{[volume(i) for i in intervals]}\n")

    eprint(f"\nMinimum interval volume: {volume_interval_min}")
    eprint(f"Sum of all overlap volumes: {volume_overlap}")
    eprint(f"Input space volume: {volume_input_space}")

    eprint("\nEstimating the percentage of volume covered …")
    matchs = [
        match(centers=centers, spreads=spreads, x=x)
        for x in st.uniform(loc=-1, scale=2).rvs(500_000,
                                                 random_state=random_state)
    ]
    # Drop the default rule entries.
    matchs = np.array(matchs)[:, 1:]
    # Count how many rules match each input.
    matchs = np.sum(matchs, axis=1)
    ratio_vol_covered = np.sum(matchs != 0) / len(matchs)
    eprint(f"Percentage of volume covered (MC approximation): "
           f"{ratio_vol_covered * 100:.1f} %")

    # d coefficients per rule.
    coefs = st.uniform(loc=-4, scale=8).rvs((n_components, dimension),
                                            random_state=random_state)

    # One intercept per rule.
    intercepts = st.uniform(loc=-4, scale=8).rvs(n_components,
                                                 random_state=random_state)

    # Noise is fixed per rule (also, assume same noise for each dimension).
    std_noises = st.gamma(a=1.0, scale=0.1).rvs(n_components,
                                                random_state=random_state)

    # One mixing coefficient per rule.
    mixing_weights = st.uniform().rvs(n_components, random_state=random_state)

    # The mixing weight of the default rule should be much smaller than any
    # other mixing weight.
    mixing_weights = mixing_weights + 0.1
    mixing_weights[0] = np.finfo(None).tiny

    X = st.uniform(loc=-1, scale=2).rvs((n, dimension),
                                        random_state=random_state)
    y = [
        output(centers=centers,
               spreads=spreads,
               coefs=coefs,
               intercepts=intercepts,
               mixing_weights=mixing_weights,
               std_noises=std_noises,
               random_state=random_state,
               x=x) for x in X
    ]

    X_test = st.uniform(loc=-1, scale=2).rvs((n, dimension), random_state=random_state)
    y_test = [
        output(centers=centers,
               spreads=spreads,
               coefs=coefs,
               intercepts=intercepts,
               mixing_weights=mixing_weights,
               std_noises=std_noises,
               random_state=random_state,
               x=x) for x in X_test
    ]


    eprint(f"\nStoring generative model and data in {npz} …")
    np.savez_compressed(npz,
                        X=X,
                        y=y,
                        X_test=X_test,
                        y_test=y_test,
                        centers=centers,
                        spreads=spreads,
                        coefs=coefs,
                        intercepts=intercepts,
                        mixing_weights=mixing_weights,
                        std_noises=std_noises)

    eprint("\nComputing match counts …")
    counts_match = np.sum(
        [match(centers=centers, spreads=spreads, x=x) for x in X], axis=0)

    eprint(f"Match counts: {counts_match}")

    stdout_csv = False
    if stdout_csv:
        X = pd.DataFrame(X).rename(columns=lambda i: f"X{i}")
        y = pd.Series(y).rename("y")
        print(pd.concat([X, y], axis=1).to_csv(index=False))

    eprint("\nChecking for data linearity by training a linear model …")
    model = make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(regressor=LinearRegression(),
                                   transformer=StandardScaler()))
    model.fit(X, y)

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
        pc.set_array(100 * random_state.random(n_components - 1))
        ax.add_collection(pc)
        ax.set_xbound(lower=-1, upper=1)
        ax.set_ybound(lower=-1, upper=1)

        plt.show()

    if dimension == 1:
        import matplotlib.pyplot as plt  # type: ignore
        X = np.linspace(-1, 1, 1000).reshape(-1, 1)
        y = []
        for x in X:
            y.append(
                output(centers=centers,
                       spreads=spreads,
                       coefs=coefs,
                       intercepts=intercepts,
                       mixing_weights=mixing_weights,
                       std_noises=std_noises,
                       random_state=random_state,
                       x=x))

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
