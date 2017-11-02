from .qa import QAEnvironment
from .synth import SynthEnvironment

ENVS = ('qa', 'synth')
QA, SYNTH = ENVS

def _get_env(env):
    if env == SYNTH:
        return SynthEnvironment
    return QAEnvironment

def create(env, opts):
    return _get_env(env)(opts)

def parse_env_opts(init_opts, remaining_opts):
    env_type = _get_env(init_opts.env)
    opts = env_type.get_opt_parser().parse_args(remaining_opts)
    for k, v in vars(init_opts).items():
        setattr(opts, k, v)
    return opts

