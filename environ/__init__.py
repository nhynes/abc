from .real import NLEnvironment
from .synth import SynthEnvironment


ENVS = ('real', 'synth')
REAL, SYNTH = ENVS


def _get_env(env):
    if env == SYNTH:
        return SynthEnvironment
    return NLEnvironment

def create(env, opts):
    """Creates the Environment appropriate for the given opts."""
    return _get_env(env)(opts)

def parse_env_opts(init_opts, remaining_opts, no_defaults=False):
    """Returns environment-specific options."""
    env_type = _get_env(init_opts.env)
    parser = env_type.get_opt_parser()
    opts = parser.parse_args(remaining_opts)
    if no_defaults:
        parser.set_defaults(**{opt: None for opt in vars(opts)})
        opts = parser.parse_args(remaining_opts)
    for k, v in vars(init_opts).items():
        if k == 'env':
            continue
        setattr(opts, k, v)
    return opts
