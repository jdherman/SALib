import argparse


def setup(parser):
    parser.add_argument(
        '-p', '--paramfile', type=str, required=True,
        help='Parameter Range File')
    parser.add_argument(
        '-r', '--result', type=str, required=True, help='Output File')
    parser.add_argument(
        '-s', '--seed', type=int, required=False, default=None,
        help='Random Seed')
    parser.add_argument(
        '--delimiter', type=str, required=False, default=' ',
        help='Column delimiter')
    parser.add_argument('--precision', type=int, required=False,
                        default=8, help='Output floating-point precision')

    return parser


def create():
    parser = argparse.ArgumentParser(
        description='Create parameter samples for sensitivity analysis')
    parser = setup(parser)

    return parser


def run_cli(cli_args, run_sample):
    parser = create()
    parser = cli_args(parser)
    args = parser.parse_args()

    run_sample(args)
