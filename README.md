# Adversarial Behavioral Cloning

Improves on the [SeqGAN](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14344)
idea by adding more reinfocement learning and GAN techniques like:
* a replay buffer
* [Consensus Optimization](https://arxiv.org/abs/1705.10461)
* [count-based exploration bonus](https://arxiv.org/abs/1611.04717)
* Proximal Policy Optimization (was not found to help, but can be found in
  [this commit](af7d921ab96e20ba75a60558e1e293b8667b4480))
* advantage normalization


## How to run

If you wish to enable Consensus Optimization (via the `--grad-reg` option), you'll need
to [patch](scripts/dx2_rnn.patch) PyTorch to allow forcing the use a
twice-differentiable RNN.

`python3 main.py` will run the project with the default options.
Output will be written to the `run/` directory.

## Shameless plug

The [em](https://github.com/nhynes/em) tool makes it really easy to twiddle
hyperparameters by tracking changes to code (no need to make everything an option!).

Just run `em run -g 0 exp_name` with your desired options and you'll find a reproducable
snapshot in `experiments/<exp_name>`!

If you want to resume from a snapshot (perhaps with different options),
use `em resume -g 0 exp_name ...`

You can also fork an experiment and its changes using `em fork`, but the quick and dirty
solution is to run `bash scripts/make_links.sh` :)
