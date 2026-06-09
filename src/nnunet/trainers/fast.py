"""
nnUNetTrainerFast — fast, reproducible base for every synthesis method.

This is Tier 0 of the trainer hierarchy.  Every method base (V26_6, SynthSeg)
and every dataset trainer inherits from it, so optimisations here propagate
everywhere automatically.

Design rule: speed *without altering nnUNet's core training logic*.  We only
touch things that are (a) pure overhead, (b) diagnostic, or (c) global compute
flags.  The forward/backward pass, loss, optimiser, sampling, validation
cadence and checkpoint selection are left exactly as upstream defines them.

What it changes vs. stock nnUNetTrainer
---------------------------------------
1. Deterministic seeding (BASE_SEED + fold) for all RNGs — reproducible data
   sampling, augmentation and weight init.  We intentionally do NOT force
   cudnn.deterministic: the entry point sets cudnn.benchmark=True for speed and
   we keep it.  Conv-level non-determinism is the accepted price of a *fast*
   seed-trainer; everything that affects the data the model sees is seeded.

2. Progress-PNG throttling.  Stock nnUNet re-renders a 30x54" seaborn figure
   *every epoch* inside on_epoch_end (LocalLogger.plot_progress_png).  Measured
   at ~790 ms/epoch on this machine — ~6.5 min of pure plotting per fold over
   500 epochs, and four folds render concurrently on the shared box.  It is
   100% redundant with the WandB curves.  We wrap the logger's plotter so it
   only renders every NNUNET_PLOT_EVERY epochs (and always on the final epoch,
   so progress.png is complete when training ends).

3. TF32 / high matmul precision.  The nnUNetv2_train entry point enables
   cudnn.benchmark but never TF32.  On Ampere (A6000) this is a free speedup
   for any fp32 matmul/conv outside autocast.  Enabled here, idempotently.

4. Architecture-plot skip.  plot_network_architecture() builds a hiddenlayer
   graph PDF at startup — slow and prone to hanging.  It is purely diagnostic
   and already skipped when torch.compile is on (the default).  We skip it
   unless NNUNET_PLOT_ARCH=1, so it never costs startup time or hangs a run.

Hooks provided for subclasses
-----------------------------
- _log_wandb_images(epoch): no-op here; method bases render their own panels.
- _aug_latency_sum_ms / _aug_latency_n: accumulate per-step augmentation
  latency (ms); logged to WandB at epoch end and reset.

Environment variables
----------------------
- NNUNET_NUM_EPOCHS            (default 500)
- NNUNET_ITERS_PER_EPOCH       (default 150)
- NNUNET_PLOT_EVERY            (default 25)  progress.png render cadence
- NNUNET_PLOT_ARCH             (default 0)   1 -> render network arch PDF
- nnUNet_compile               (upstream)    1/0 to force torch.compile on/off
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

BASE_SEED = 12345

# WandB sample-image panels: log every epoch for the first 10, then every N.
_WANDB_IMG_EPOCHS_ALWAYS = set(range(10))
_WANDB_IMG_EVERY_N = 10


def _enable_tf32() -> None:
    """Enable TF32 / high matmul precision (free Ampere speedup). Idempotent."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


class _ChannelsLastInstanceNorm3d(torch.nn.Module):
    """
    channels_last-native drop-in for nn.InstanceNorm3d (affine, no running stats).

    Identical math to torch's InstanceNorm3d — per-(N,C) mean/var over the spatial
    dims, then affine — but it does NOT reshape (N,C,…)→(1,N·C,…) the way
    F.instance_norm does.  That reshape is what breaks the channels_last memory
    format (forcing huge copies), which in turn forces cuDNN to insert NCHW↔NHWC
    transposes around every conv (~16.5% of GPU time, measured).  With this
    norm the whole network runs channels_last_3d, the transposes vanish, and
    torch.compile fuses the var/mean/affine elementwise into a few kernels.
    """
    def __init__(self, src: torch.nn.Module):
        super().__init__()
        self.eps = src.eps
        self.affine = src.affine
        if src.affine:
            self.weight = src.weight
            self.bias = src.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stats in fp32 to match cuDNN's InstanceNorm (which accumulates mean/var
        # in float even for half input) — keeps the result numerically identical.
        xf = x.float()
        var, mean = torch.var_mean(xf, dim=(2, 3, 4), keepdim=True, unbiased=False)
        xf = (xf - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            xf = xf * self.weight.float().view(1, -1, 1, 1, 1) + self.bias.float().view(1, -1, 1, 1, 1)
        return xf.to(x.dtype)


def _swap_instancenorm3d(module: torch.nn.Module) -> int:
    """Recursively replace nn.InstanceNorm3d with the channels_last-native version."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.InstanceNorm3d):
            setattr(module, name, _ChannelsLastInstanceNorm3d(child))
            n += 1
        else:
            n += _swap_instancenorm3d(child)
    return n


def _to_channels_last_pre_hook(_module, args):
    """forward_pre_hook: feed the network channels_last_3d input (5-D only)."""
    if args and torch.is_tensor(args[0]) and args[0].dim() == 5:
        return (args[0].contiguous(memory_format=torch.channels_last_3d),) + tuple(args[1:])
    return args


class nnUNetTrainerFast(nnUNetTrainer):
    """
    Drop-in nnUNetTrainer with general speed best-practices.

    All method-specific bases (V26_6, SynthSeg) and dataset trainers inherit
    from this rather than directly from nnUNetTrainer, so every optimisation
    added here applies to all of them at once.
    """

    def initialize(self) -> None:
        # Per-fold AND per-rank seed.  Under DDP (num_gpus > 1) every rank must
        # draw DIFFERENT augmentation/sampling RNG, otherwise all ranks compute
        # identical batches and data-parallelism degenerates to batch-1 diversity.
        # InstanceNorm (this network's norm) is per-sample, so DDP batch-1-per-GPU
        # is mathematically equivalent to single-GPU batch-N — no accuracy change.
        seed = BASE_SEED + 1000 * self.fold + self.local_rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        _enable_tf32()

        # channels_last_3d: defer nnUNet's torch.compile so we can swap the
        # InstanceNorm3d layers for a channels_last-native version FIRST, then
        # convert to channels_last and compile.  This removes the ~16.5% cuDNN
        # NCHW<->NHWC transpose tax (measured) and lets compile fuse the norm.
        # Opt-in (NNUNET_CHANNELS_LAST=1): removes the cuDNN transpose tax but
        # swaps InstanceNorm3d for a math-identical reimplementation, which
        # introduces fp16-rounding-level activation deltas (~0.2% mean-relative,
        # same scale as run-to-run cudnn.benchmark nondeterminism).  ~1.16× on the
        # network, ~1.07× on the full epoch.  Off by default to keep bit-parity.
        use_cl = (os.environ.get("NNUNET_CHANNELS_LAST", "0").lower() in ("1", "true", "t")
                  and self.device.type == "cuda")
        want_compile = self._do_i_compile() if use_cl else False
        prev_compile_env = os.environ.get("nnUNet_compile")
        if use_cl:
            os.environ["nnUNet_compile"] = "0"   # defer; compiled after the swap

        # MUST precede super().initialize(): configure_optimizers() builds the PolyLR
        # scheduler with max_steps=self.num_epochs. Setting num_epochs *after* left the
        # scheduler at nnUNet's default (1000), so resuming past 1000 produced a
        # negative→complex LR and crashed. Set it here so the scheduler uses the real value.
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", "500"))
        self.num_iterations_per_epoch = int(os.environ.get("NNUNET_ITERS_PER_EPOCH", "150"))
        # Optional LR override (also must precede super().initialize() — feeds both the
        # SGD optimizer and the PolyLR horizon). Use to cap the restart LR on resume,
        # e.g. resuming at epoch E with horizon N, LR_start = initial_lr*(1-E/N)^0.9.
        if os.environ.get("NNUNET_INITIAL_LR"):
            self.initial_lr = float(os.environ["NNUNET_INITIAL_LR"])

        super().initialize()

        if use_cl:
            if prev_compile_env is None:
                os.environ.pop("nnUNet_compile", None)
            else:
                os.environ["nnUNet_compile"] = prev_compile_env
            net = self.network.module if self.is_ddp else self.network
            n_swapped = _swap_instancenorm3d(net)
            net.to(memory_format=torch.channels_last_3d)
            net.register_forward_pre_hook(_to_channels_last_pre_hook)
            if want_compile:
                self.network = torch.compile(self.network)
            self.print_to_log_file(
                f"[Fast] channels_last_3d on: {n_swapped} InstanceNorm3d swapped, "
                f"compile={'on' if want_compile else 'off'}")

        self._aug_latency_sum_ms = 0.0
        self._aug_latency_n = 0

        self._install_progress_plot_throttle()

    # ------------------------------------------------------------------ #
    # Speed overrides (no core-logic change)
    # ------------------------------------------------------------------ #
    def _install_progress_plot_throttle(self) -> None:
        """
        Wrap self.logger.plot_progress_png so it renders only every
        NNUNET_PLOT_EVERY epochs (and on the last epoch).  Stock nnUNet calls
        it every epoch from on_epoch_end — ~790 ms each, redundant with WandB.

        Operates on the logger instance only; nnUNet's library code is untouched.
        At the call site (parent on_epoch_end) self.current_epoch is the
        just-finished epoch — it is incremented *after* the plot call.
        """
        every = int(os.environ.get("NNUNET_PLOT_EVERY", "25"))
        if every <= 1:
            return  # keep stock per-epoch behaviour

        original = self.logger.plot_progress_png
        trainer = self

        def throttled(output_folder):
            ep = trainer.current_epoch
            if ep % every == 0 or ep >= trainer.num_epochs - 1:
                original(output_folder)

        self.logger.plot_progress_png = throttled

    def plot_network_architecture(self) -> None:
        """Skip the hiddenlayer architecture PDF unless NNUNET_PLOT_ARCH=1.

        It is diagnostic-only, slow, and prone to hanging.  Upstream already
        skips it when torch.compile is enabled (the default).
        """
        if os.environ.get("NNUNET_PLOT_ARCH", "0").lower() in ("1", "true", "t"):
            super().plot_network_architecture()

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def on_epoch_end(self) -> None:
        # Save epoch BEFORE super(): nnUNet increments self.current_epoch inside
        # on_epoch_end(), so any wandb.log(..., step=N) called after would use N+1.
        epoch = self.current_epoch
        super().on_epoch_end()

        # Only rank 0 logs to WandB under DDP.
        if self.local_rank != 0:
            self._aug_latency_sum_ms = 0.0
            self._aug_latency_n = 0
            return

        if self._aug_latency_n > 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {"train/aug_latency_ms": self._aug_latency_sum_ms / self._aug_latency_n},
                        step=epoch,
                    )
            except Exception:
                pass
        self._aug_latency_sum_ms = 0.0
        self._aug_latency_n = 0

        if epoch in _WANDB_IMG_EPOCHS_ALWAYS or epoch % _WANDB_IMG_EVERY_N == 0:
            try:
                self._log_wandb_images(epoch=epoch)
            except Exception as e:
                print(f"[WandB] image logging failed at epoch {epoch}: {e}")

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """Override in method-specific subclasses to log a sample image panel."""
