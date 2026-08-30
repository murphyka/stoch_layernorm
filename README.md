# Tracing distinguishability through transformer processing with stochastic LayerNorm
[arxiv link](https://arxiv.org/abs/)

This repository contains training and analysis code for treating each LayerNorm read in a transformer as a rate-limited channel. Each read is modeled as a von Mises--Fisher (vMF) random variable on the sphere induced by LayerNorm. The concentration parameter `kappa` controls the read precision, and the network is trained under a fixed total rate budget shared across all taps, learning how to allocate that budget across the model.

The accompanying analysis measures how distinguishability between inputs propagates through these noisy reads, including through the Q, K, and V projections of individual attention heads.

The same channel construction is used for two architectures:

* ViT-Small on ImageNet-1k
* GPT-2-small on OpenWebText

## Repository layout

### Channel and calibration

| file           | description                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vmf_utils.py` | vMF rate/noise calibration utilities: closed-form `rate = KL(vMF(kappa) \|\| uniform)`, mean resultant length `A_p`, the matched noise scale `sigma = sqrt(1/rho^2 - 1)`, pairwise Bhattacharyya coefficient, and a Wood/Ulrich sampler. Running `python vmf_utils.py` produces a self-validation table. |

### Training

| file                           | description                                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_stoch_layernorm_vit.py` | ViT trainer using timm ViT-Small on ImageNet. Implements the noisy-LayerNorm channel, softmax rate-budget allocation, and per-step budget ramp.               |
| `train_stoch_layernorm_gpt.py` | GPT-2 trainer using the same channel and allocation policy, adapted to causal language modeling with packed token blocks and shifted CE / chunked shifted KL. |
| `prepare_lm_data.py`           | Pre-tokenizes a text corpus into flat `uint16` memmaps (`train.bin`, `val.bin`, `meta.json`).                                                                 |

### Bhattacharyya analysis and per-head Q/K/V

| file                            | description                                                                                                                                                                                                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bhat_qkv.py`                   | Core estimator for the exact pushforward Bhattacharyya coefficient of a vMF posterior through a head's Q/K/V projection, using an invertible whitening of the output projection. This does not use Gaussian moment matching or KDE. A derivation is given in the `bc_projected` docstring. |
| `qkv_bc_gpt.py`                 | GPT-2 driver for per-head Q/K/V BC curves across layers and heads from a trained checkpoint. Also provides `load_run`, which reconstructs the model, policy, and per-tap `kappa` values from a checkpoint.                                                                                                                                                                                               |
| `qkv_bc_vit.py`                 | ViT driver for per-head Q/K/V BC maps comparing a base image with a designed patch perturbation.                                                                                                                                                                                           |
| `random_projection_baseline.py` | Random-projection baseline measuring how much of a tap's log-BC is preserved by an uninformative read of the same dimensionality.                                                                                                                                                          |
| `crn_perturb_vit.py`            | Common-random-numbers paired-draw perturbation analysis for ViTs. Used by `qkv_bc_vit.py`.                                                                                                                                                                                                 |
| `bc_bridge.py`                  | Bounded-variance Monte Carlo estimator of BC between two vMF mixtures, for cases not covered by the exact pushforward calculation.                                                                                                                                                         |

## Implementation details

### Tap indices are not in depth order

Taps are assigned according to `named_modules()` registration order. In both timm's ViT and HuggingFace's GPT-2, this places the final normalization layer (`norm` / `ln_f`) at index `0`, followed by the block norms.

As a result, a policy vector indexed by `module.idx` should not be interpreted as being in model depth order. `load_run` handles this by indexing `kappa` using `module.idx` while obtaining depth order from a separate `named_modules()` traversal. New analyses should preserve this distinction rather than assuming that `kappas[i]` corresponds to block `i`.

### The kappa grid imposes a maximum noise scale

`build_rate_sigma_maps` constructs a monotone grid beginning at `kappa_lo`, and interpolation is clamped to this range. The lower end of the grid therefore determines the largest representable per-tap noise:

* `sigma <= 76.6` for `p=383`
* `sigma <= 10.1` for `p=767`

This limit is reached in the strongest-compression runs, where many taps lie exactly at the cap. These values should therefore be interpreted as censored rather than as resolved noise estimates.

The usable lower bound on `kappa` depends on dimensionality because the Bessel term underflows to zero in double precision at sufficiently large `p`. `train_stoch_layernorm_gpt.safe_kappa_lo` determines the floor by bisection.

## Usage

Calibration self-check, requiring neither GPU nor dataset:

```bash
python vmf_utils.py
```

### Training

`--sigma_g` is the dimension-independent noise parameter used to set the total rate budget. The budget ramp is defined in optimization steps, so `--noise_ramp_epochs` retains the same interpretation when the total number of training epochs changes.

```bash
# ViT: fine-tune from timm pretrained weights while distilling from the frozen clean model
python train_stoch_layernorm_vit.py --arm finetune --sigma_g 1.0 --model vit_small_patch16_224 \
    --data /path/to/imagenet --val_subdir val_split --epochs 25 --batch_size 512 \
    --distill --teacher_pretrained --noise_ramp_epochs 6 --ramp_shape geom --out runs/vit_s_sigma1

# GPT-2: apply the same channel to packed OpenWebText blocks
python prepare_lm_data.py --data_name /path/to/openwebtext --out /path/to/tokenized
python train_stoch_layernorm_gpt.py --arm finetune --sigma_g 1.0 --data_name /path/to/tokenized \
    --epochs 25 --batch_size 32 --block_size 512 --distill \
    --noise_ramp_epochs 6 --ramp_shape geom --out runs/gpt2_sigma1
```

### Q/K/V Bhattacharyya analysis

```bash
python qkv_bc_gpt.py --ckpt runs/gpt2_sigma1/ckpt.pt
python qkv_bc_vit.py  --ckpt runs/vit_s_sigma1/ckpt.pt
python random_projection_baseline.py --ckpt runs/gpt2_sigma1/ckpt.pt
```

The `--ckpt` defaults in the driver scripts refer to local paths used for the original experiments. Pass an explicit checkpoint path for other runs.

## Caveats

* **Q/K/V analyses include upstream noise.** The probes estimate CRN-conditioned posterior means under the actual stochastic forward process rather than substituting a clean deterministic forward pass. The clean-pass version is not equivalent and can give substantially different results.

* **Per-head retention should be compared with the random-projection baseline.** A random read with the same output dimensionality can preserve a substantial fraction of the apparent distinguishability. `random_projection_baseline.py` is included to quantify this effect.

* **`vmf_utils.log_bc_mc` is not recommended for mixture-of-posteriors BC.** A split-half identity test, in which two samples from the same distribution should give BC near 1, fails at many taps for structural reasons rather than simply from insufficient Monte Carlo samples. For current analyses, use the exact pushforward calculation in `bhat_qkv`, the paired-CRN approach in `crn_perturb_vit`, or `bc_bridge` for genuine mixtures. `log_bc_mc` remains as a variance baseline for `bc_bridge`. The `sample_mixture` and `mixture_log_density` utilities are still used by the Q/K/V estimator.

* **The ViT and GPT-2 trainers differ in a few architecture-specific details.** They use the same channel implementation, but differ in the rule used to choose the lower end of the `kappa` grid, the inherited LayerNorm epsilon (`1e-6` in timm versus `1e-5` in HuggingFace), and some historical details of weight-decay grouping. Both currently exclude LayerNorm gains and biases from weight decay; the ViT implementation additionally excludes `pos_embed` and `cls_token`, following timm's training recipe.

## Requirements

`torch`, `numpy`, `scipy`, `timm`, `transformers`, `datasets`, `torchvision`, `matplotlib`, and `Pillow`.

Training experiments used bf16 autocast on A100 GPUs.

## License

MIT. See [LICENSE](LICENSE).
