import warnings
from typing import Dict, Optional, Union

import numpy as np
from scipy.stats import qmc

from . import common_args
from ..util import scale_samples, read_param_file, compute_groups_matrix, _check_groups


def sample(
    problem: Dict,
    N: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    skip_values: int = 0,
    base_sequence: Optional[np.ndarray] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    """Generates model inputs using Saltelli's extension of the Sobol' sequence.

    The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
    to generate uniform samples of parameter space.
    The general approach is described in [1].

    Returns a NumPy matrix containing the model inputs using Saltelli's
    sampling scheme.

    Saltelli's scheme reduces the number of required model runs from ``N(2D+1)`` to
    ``N(D+1)`` (see [2]).

    If `calc_second_order` is False, the resulting matrix has ``N * (D + 2)``
    rows, where ``D`` is the number of parameters.

    If `calc_second_order` is `True`, the resulting matrix has ``N * (2D + 2)``
    rows.

    These model inputs are intended to be used with
    :func:`SALib.analyze.sobol.analyze`.

    Notes
    -----
    The initial points of the Sobol' sequence has some repetition (see Table 2
    in Campolongo [3]__), which can be avoided by scrambling the sequence.

    Another option, not recommended and available for educational purposes,
    is to use the `skip_values` parameter.
    Skipping values reportedly improves the uniformity of samples.
    But, it has been shown that naively skipping values may reduce accuracy,
    increasing the number of samples needed to achieve convergence
    (see Owen [4]__).

    Parameters
    ----------
    problem : dict,
        The problem definition.
    N : int
        The number of samples to generate.
        Ideally a power of 2 and <= `skip_values`.
    calc_second_order : bool, optional
        Calculate second-order sensitivities. Default is True.
    scramble : bool, optional
        If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
        Default is True.
    skip_values : int, optional
        Number of points in Sobol' sequence to skip, ideally a value of base 2.
        It's recommended not to change this value and use `scramble` instead.
        `scramble` and `skip_values` can be used together.
        Default is 0.
    base_sequence : array_like
        If `base_sequence` is None the Sobol' low discrepancy sequence is used
        as a base to generate the sample needed for the Sobol' indices method.
        Otherwise, provide a sample of shape ``(2 * D, N)``.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` generator is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used. Default is None.

    References
    ----------
    1. Sobol', I.M., 2001.
       Global sensitivity indices for nonlinear mathematical models and
       their Monte Carlo estimates.
       Mathematics and Computers in Simulation,
       The Second IMACS Seminar on Monte Carlo Methods 55, 271-280.
       https://doi.org/10.1016/S0378-4754(00)00270-6

    2. Saltelli, A. (2002).
       Making best use of model evaluations to compute sensitivity indices.
       Computer Physics Communications, 145(2), 280-297.
       https://doi.org/10.1016/S0010-4655(02)00280-1

    3. Campolongo, F., Saltelli, A., Cariboni, J., 2011.
       From screening to quantitative sensitivity analysis.
       A unified approach.
       Computer Physics Communications 182, 978-988.
       https://doi.org/10.1016/j.cpc.2010.12.039

    4. Owen, A. B., 2020.
       On dropping the first Sobol' point.
       arXiv:2008.08051 [cs, math, stat].
       Available at: http://arxiv.org/abs/2008.08051
       (Accessed: 20 April 2021).
    """
    D = problem["num_vars"]
    groups = _check_groups(problem)

    # Create base sequence - could be any type of sampling
    # sample Sobol' sequence
    if base_sequence is None:
        qrng = qmc.Sobol(d=2 * D, scramble=scramble, seed=seed)

        # fast-forward logic
        if skip_values > 0 and isinstance(skip_values, int):
            M = skip_values
            if not ((M & (M - 1) == 0) and (M != 0 and M - 1 != 0)):
                msg = f"""
                Convergence properties of the Sobol' sequence is only valid if
                `skip_values` ({M}) is a power of 2.
                """
                warnings.warn(msg, stacklevel=2)

            # warning when N > skip_values
            # see https://github.com/scipy/scipy/pull/10844#issuecomment-673029539
            n_exp = int(np.log2(N))
            m_exp = int(np.log2(M))
            if n_exp > m_exp:
                msg = (
                    "Convergence may not be valid as the number of "
                    "requested samples is"
                    f" > `skip_values` ({N} > {M})."
                )
                warnings.warn(msg, stacklevel=2)

            qrng.fast_forward(M)
        elif skip_values < 0 or not isinstance(skip_values, int):
            raise ValueError("`skip_values` must be a positive integer.")

        base_sequence = qrng.random(N)
    else:
        target_shape = (2 * D, N)
        if base_sequence.shape != target_shape:
            raise ValueError(
                f"Incorrect 'base_sequence' shape {base_sequence.shape}. "
                f"It must be {target_shape}"
            )

    if not groups:
        Dg = problem["num_vars"]
    else:
        G, group_names = compute_groups_matrix(groups)
        Dg = len(set(group_names))

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.zeros([(Dg + 2) * N, D])

    index = 0

    for i in range(N):
        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(Dg):
            for j in range(D):
                if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                    saltelli_sequence[index, j] = base_sequence[i, j + D]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (
                        groups and group_names[k] == groups[j]
                    ):
                        saltelli_sequence[index, j] = base_sequence[i, j]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j + D]

                index += 1

        # Copy matrix "B"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j + D]

        index += 1

    saltelli_sequence = scale_samples(saltelli_sequence, problem)
    return saltelli_sequence


def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    -------
    Updated argparse object
    """
    parser.add_argument(
        "--max-order",
        type=int,
        required=False,
        default=2,
        choices=[1, 2],
        help="Maximum order of sensitivity indices to calculate",
    )

    parser.add_argument(
        "--scramble",
        type=int,
        required=False,
        default=True,
        help="Use scrambled sequence",
    )

    parser.add_argument(
        "--skip-values",
        type=int,
        required=False,
        default=None,
        help="Number of sample points to skip (default: next largest power of"
        " 2 from `samples`). Not recommended (use `scramble` instead).",
    )

    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(
        problem,
        args.samples,
        calc_second_order=(args.max_order == 2),
        scramble=args.scramble,
    )
    np.savetxt(
        args.output,
        param_values,
        delimiter=args.delimiter,
        fmt="%." + str(args.precision) + "e",
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
