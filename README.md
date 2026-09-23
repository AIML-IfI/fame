# FAME: Feature Activation Map Explanation

Reference implementation of

> Xinyi Zhang and Manuel Günther. **FAME: Feature Activation Map Explanation on
> Image Classification and Face Recognition.** CVPR Workshops (XAI4CV), 2026.

FAME reinterprets the LOTS adversarial attack as an attribution method: it
iteratively perturbs the input with normalized gradients until the network
output reaches a chosen target, then reads the accumulated perturbation
`|x̄ − x|` as a pixel-level explanation.

## Layout

```
fame/
├── fame/
│   ├── optimizer.py       the optimization loop (Eq. 3) -- the only copy
│   ├── losses.py          L_a, L_cls, L_+, L_- (Eq. 5, 6)
│   ├── attribution.py     perturbation -> grayscale -> blur -> normalize (Eq. 4)
│   ├── explain.py         the three task entry points + FameConfig
│   ├── visualization.py   heatmaps and overlays
│   ├── models/            classifier zoo, IResNet/AdaFace, input wrappers
│   ├── data/              ImageNet subset, verification pair loader
│   ├── baselines/         Grad-CAM family, FGGB, CorrRISE
│   └── metrics/           IoU, ROAD-Delete, deletion/insertion, EER
│   ├── benchmark.py       timing and peak-memory measurement
│   ├── runlog.py          per-run log, stage timings and manifest
├── scripts/               command line entry points
├── tests/                 pytest suite, runs on CPU in ~15s
└── protocols/             imagenet.csv, imagenet_sub.csv
```

The optimization loop is the procedure Rozsa et al. published as the LOTS
adversarial attack. Since FAME's contribution is the reinterpretation, the
identifiers use the FAME name (`from fame import fame`) and the attribution to
LOTS lives in the docstrings and the references. Modules are never named after
the function they export -- a submodule overwrites the same-named attribute on
its package whenever it is imported, so `fame/fame.py` would make `fame.fame`
resolve to either the module or the function depending on import order. See
also `optimizer.py`, `baselines/perturbation.py` and `baselines/gradient.py`.

## Install

```bash
pip install -r requirements.txt
```

The scripts add the repository root to `sys.path` themselves, so they run from
a fresh checkout with no PYTHONPATH and no install. `pip install -e .` works
too if you want to import `fame` from elsewhere.

## Usage

```python
import torch
from fame import FameConfig, explain_classification, explain_verification
from fame.models import build_classifier

model = build_classifier("ResNet101", device="cuda")   # takes [0, 1] images
config = FameConfig(iterations=500, step_size=1/255)

attribution = explain_classification(model, images, targets, config)  # (B, 1, H, W)
```

For a verification pair, both maps come from one call:

```python
maps = explain_verification(face_model, probe, gallery, config)
e_plus, e_minus = maps["similar"], maps["dissimilar"]
```

### Command line

```bash
# ImageNet attributions, then the CAM baselines, then Tab. 1
python scripts/run_fame_classification.py --images-root $VAL --synset-mapping $SYNSETS \
    --output results/FAME --models ResNet50 ConvNeXt_Tiny
python scripts/run_baselines_classification.py --images-root $VAL --synset-mapping $SYNSETS \
    --output results
python scripts/eval_classification.py --images-root $VAL --synset-mapping $SYNSETS \
    --attributions results --output results/classification_metrics.csv

# Face verification attributions, then Tab. 2
python scripts/run_fame_verification.py --images-root $CROPS --protocol-dir protocols/ARface \
    --checkpoint $ADAFACE --model Adaface_ir_101 --dataset ARface --method FAME \
    --output results
python scripts/eval_verification.py --images-root $CROPS --protocol-dir protocols/ARface \
    --checkpoint $ADAFACE --model Adaface_ir_101 --dataset ARface \
    --attributions results --methods FAME GradCAM CorrRISE FGGB \
    --output results/ARface_metrics.csv

# Receptive fields of all 49 feature map locations (Sec. 4.1, Fig. 7 and 8)
python scripts/run_fame_featuremap.py --image sample.JPEG --model ResNet101 \
    --output results/featuremap --grid --mark-receptive-field
```

### Face recognition scores and the operating point

```bash
# Per-pair scores plus the EER threshold table, needed before Tab. 2
python scripts/compute_verification_scores.py --images-root $CROPS \
    --protocol-dir $PROTOCOLS --datasets ARface SCface CFP \
    --checkpoints Adaface_ir_101=$CKPT --output results/scores

# Fig. 4: deletion and insertion curves from the metrics CSV
python scripts/plot_curves.py --task verification \
    --metrics results/CFP_metrics.csv --output results/figures \
    --clean-accuracy 99.86
```

### Supplemental material

```bash
# Sec. A / Fig. 6: sensitivity to iterations, step size and blur
python scripts/run_parameter_sweep.py --image sample.JPEG --model ResNet50 \
    --sweep iterations step_size blur --output results/sweeps

# Sec. B / Tab. 3(a): FAME runtime against iteration count
python scripts/run_runtime_benchmark.py --table a --images-root $VAL \
    --synset-mapping $SYNSETS --limit 1000 --output results/runtime_a.csv

# Sec. B / Tab. 3(b): runtime of every method on CFP-FP
python scripts/run_runtime_benchmark.py --table b --crops-root $CROPS \
    --protocol-dir protocols/CFP --checkpoint $ADAFACE --face-protocol 01FP \
    --output results/runtime_b.csv

# Sec. C / Fig. 9 to 12: method-by-network comparison sheets
python scripts/plot_comparison_grid.py --task verification --crops-root $CROPS \
    --protocol-dir protocols/CFP --dataset CFP --protocol-name 01FP \
    --attributions results --pairs 0 1 2 --output results/figures
```

Two details the supplemental is explicit about, and which the scripts follow.
The runtime benchmark defaults to `--mode sequential`, because the reference
implementations of CorrRISE and FGGB process one image at a time and batching
only FAME would not be a fair comparison; `--mode batched` measures the
parallel path instead. The parameter sweep disables early stopping, since
otherwise the shorter runs would not take the number of steps they are labelled
with.

Two values could not be read off the paper and are inferred, so check them
against your figures: the iteration counts of Fig. 6(a) and Tab. 3(a) are taken
as `[1, 25, 50, 75, 100, 200, 300, 400, 500]`, which matches the nine rows of
the table and its roughly linear growth of ~19 s per iteration per 1000 images;
and Sec. 4.4 describes the chosen blur as "a kernel size of 7.7" while Fig. 5's
caption makes 7.7 the standard deviation, so the code follows your
implementation with kernel 49 and sigma 7.7. Both are `--` arguments.

Runtime scales linearly with `--iterations`. The supplemental material reports
that maps stabilize between 75 and 200 iterations, so `--iterations 100` is a
reasonable setting during development.

## What changed relative to the original scripts

Behavioral fixes, in rough order of impact on results:

1. **Attributions are saved as finished grayscale maps, not jet-coloured RGB.**
   `save_results` wrote the colormapped `heatmaps` to `.npy` while discarding
   the `normalized_tensor` sitting right next to it, so the evaluation blurred
   and ranked a colormap. It now saves the grayscale map — already blurred and
   normalized to [0, 1] — and the evaluation loads and ranks it directly, with
   no post-processing to repeat. `--save-overlay` writes the colored PNG
   separately for figures.
2. **`Adaface_model.forward` discarded its own normalization.** It computed
   `out = self.norm(x)` and then `out = self.backbone(x)`, feeding [0, 1] pixels
   to a network trained on [-1, 1].
3. **Gradient normalization is per sample.** `torch.max(torch.abs(gradient))`
   over a whole batch let one image set the step size for the others. Identical
   for batch size 1, which is what the original loops used.
4. **CAM target layers are resolved per architecture.** The hard-coded
   `model.features[-1][-1].block[-1]` is a ConvNeXt path and fails on ResNet and
   VGG.
5. **Bounding boxes use a single resize scale.** `Resize(232)` scales by the
   shorter side and keeps the aspect ratio. `adjust_bbx` scales the two axes
   independently, as if resizing to a square 232x232, which shifts the box by
   up to 23 pixels on a 500x375 image and changes IoU. `--original-boxes`
   reproduces the old convention for comparison against previously reported
   numbers.
6. **ROAD averages over the actual number of images** instead of a hard-coded
   5000. It still blacks pixels out before normalization, as `ImgNet` does, so
   removed pixels land at `-mean/std`; the neighborhood imputation of Rong et
   al. is available via `--road-imputation linear` but is not the default,
   since the original does no imputation.
7. **`L_cls` backpropagates the logit, not `|logit|`.** An L1 loss towards zero
   flips the update direction whenever the logit is negative.
8. **FGGB differentiates the L2-normalized embedding**, as `code_fggb.py` does
   by calling `_normalize_embedding` before `f_a[0, k].backward()`.
   Differentiating the raw embedding gives different maps, since normalizing
   couples the dimensions.
9. **FGGB's threshold stays divided by the embedding size.** The v_i sum to the
   cosine similarity, so each is on the order of `cos / D`. `code_fggb.py`
   correctly uses `v_i - theta / D`; subtracting a whole EER threshold from
   every v_i would leave e_+ empty and e_- covering the image.
   `per_dimension=True` exposes that variant for comparison only.
10. **CorrRISE correlates ranked scores**, as `Corrise.py` does, which makes it
   Spearman rather than Pearson. Ranking the binary masks is a no-op, so only
   the score side matters. `rank_based=False` gives the Pearson version.
   Mask patches are also drawn from `[0, H - patch_size]` so they fit entirely
   inside the image, and the default is 3 patches of 30x30, matching the
   `generate_mask_set(..., num_patches=3)` call. `masks=` accepts a shared set,
   which is how the original reuses one set for every pair.
11. **Baselines keep the smoothing they originally had.** `code_fggb.py` blurs
    with kernel 25 and sigma 5 before saving; `Corrise.py` saves raw
    correlations. Applying each at generation time keeps the evaluation free of
    per-method special cases.
12. **The grayscale conversion happens before the subtraction.** The original
    `get_difference` grays both images and then subtracts, so the absolute
    value sits between two linear steps; graying the difference instead gives a
    completely different map. This one was worth catching -- the two differ by
    up to 0.52 on a map whose range is [0, 1].
13. **The EER sweep runs downwards.** `compute_eer_threshold` builds its
    candidates with `np.unique(scores)[::-1]`, and `argmin` returns the first
    minimum, so tied thresholds resolve to the largest. Sweeping upwards
    returns a different operating point on ties, which coarse score
    distributions produce readily.
14. **One set of luma weights.** The FAME path went through
   `transforms.Grayscale` (0.2989, 0.587, 0.114) while the FGGB evaluation used
   its own `to_gray` with 0.299; `to_grayscale` now matches torchvision, which
   is what the FAME maps were built with.

Deliberately left as it was: **LOTS still takes its 1/255 step in the same
space as before.** For image classification that means the mean/std normalized
tensor, so the step is worth roughly 1.1 gray levels and differs slightly per
color channel; `--input-space pixel` switches to perturbing the [0, 1] image
if you ever want to compare. Face models normalize with a uniform 0.5/0.5, so
the distinction does not arise there.

Consolidation:

- `eval_delete_{fame,cam,fggb,corr}.py` and `eval_insert_{fame,cams,fggb,corr}.py`
  plus `eval_auc.py` → `scripts/eval_verification.py`. The eight files differed
  only in a filename, a post-processing step on load, and one comparison
  operator.
- The eighteen `cam/cam_<dataset>_<target>[_partial][_adaface].py` files →
  `fame/baselines/cam.py`, with dataset and target as arguments.
- `adaface.py` and `adaface_norm.py` were byte-identical except for a trailing
  L2 normalization → one backbone plus an `l2_normalize` flag.
- Three copies of the blur-and-normalize routine with two different kernel
  settings (49/7.7 and 25/5) → `fame.attribution.fame_attribution`.
- Hard-coded `/local/scratch/...` and `/srv/scratch_rolf/...` paths and
  `cuda:N` device strings → command line arguments.

`fame/data/pairs.py` is a port of `Pairs` from `utils.py`, reading the `G`, `P`
and `T` columns. The RGB-to-BGR swap is a constructor flag rather than part of
the dataset, matching the original split where only the AdaFace code paths
called `cv2.cvtColor`; it defaults to on, since every model in the paper is an
AdaFace model.

Reconstructed, because it was missing from the original drop:

- `explain_feature_map` and `scripts/run_fame_featuremap.py`: the driver for the
  feature map experiments (Sec. 4.1, Fig. 1, 7 and 8) was lost. Two details are
  confirmed and pinned by tests: the channel reduction in `L_a` is the sum of
  absolute values, matching the L1 norm in Eq. (5), and the experiment runs on
  the ResNets, whose `layer4` output is 7x7 at 224x224, giving 49 attribution
  maps per image that tile into a 7x7 sheet. `--batch-size 49` does a whole map
  in one pass, and the script warns if the chosen architecture gives a different
  shape (VGG19 gives 14x14). Iteration count and blur follow `FameConfig`; pass
  `--iterations` if the published figures used another value.

  Worth noting for the ResNets specifically: `layer4[-1]` ends in a ReLU, so
  a[k] has no negative channels and the absolute sum coincides with a plain sum
  there. The distinction only bites on architectures that can output negative
  activations.
- `scripts/run_fame_verification.py`: the archive contained the evaluation of
  `Results_fame_cos/` and `Results_fame_1-cos/` but not the script that produced
  them, so the generation step is written from Eq. (6).

## What reproduces, and what does not

| Paper item | Script | Status |
|---|---|---|
| Fig. 1, 7, 8 — feature map receptive fields | `run_fame_featuremap.py` | driver lost; channel reduction and 7x7 grid confirmed, iteration count assumed |
| Tab. 1 — IoU and ROAD on ImageNet | `run_fame_classification.py`, `run_baselines_classification.py`, `eval_classification.py` | protocol CSV shipped (5000 images, 5 per class); needs ImageNet val |
| Fig. 9 — IC method comparison | `plot_comparison_grid.py --task classification` | yes |
| Tab. 2 — deletion/insertion on AR Face, SCface, CFP | `compute_verification_scores.py`, `run_fame_verification.py`, `eval_verification.py` | needs the datasets, the AdaFace checkpoints and the protocol CSVs, none of which are redistributable |
| Fig. 3, 10-12 — FR method comparison | `plot_comparison_grid.py --task verification` | same data requirement |
| Fig. 4 — deletion/insertion curves | `plot_curves.py` | yes, from the metrics CSV |
| Fig. 5, 6 — blur, iteration and step size sweeps | `run_parameter_sweep.py` | yes; the swept values in Fig. 6 are inferred |
| Tab. 3 — runtime | `run_runtime_benchmark.py` | yes; wall-clock numbers are machine specific |

Three things are needed that this repository cannot contain: the ImageNet
validation set, the three face datasets with their aligned crops and protocol
files, and the AdaFace checkpoints. Everything downstream of those is here.

Two settings could not be recovered and are assumed rather than confirmed: the
iteration count for the feature map figures, and the exact swept values behind
Fig. 6 and Tab. 3(a). Both are command line arguments.

No number in the paper has been reproduced end to end here, because the
verification was done against tiny randomly initialized networks and synthetic
data rather than the real models. Given that the audit turned up deviations as
large as 0.52 on a [0, 1] attribution map, treat the tables as needing a rerun
rather than a spot check.

## Logs

Every script writes to `<output>/logs/<name>_<timestamp>.{log,json}` as well as
to the console, so a result directory explains itself later:

```
09:08:11 INFO    start fame_classification
09:08:11 INFO    command: scripts/run_fame_classification.py --models ResNet50 VGG19 ...
09:08:11 INFO      torch: 2.13.0+cu130
09:08:11 INFO      gpu: NVIDIA A100-SXM4-40GB
09:08:11 INFO    [ResNet50] start
10:41:53 INFO    [ResNet50] done in 1h 33m 42s, peak 4.21 GB
10:41:53 INFO    [ResNet50] 5000 items, 1.1244 s/item
...
12:15:30 INFO    completed in 3h 07m 19s (total 11239.11 s)
```

The JSON manifest beside it records the full argument list, the git commit
(with `-dirty` when the tree has uncommitted changes), the machine, and the
per-stage seconds and peak memory. That is what makes a runtime number
attributable to a specific configuration on a specific host — the runtime
benchmark of Tab. 3 in particular is meaningless without it.

Runs that crash still write their manifest, with `"status": "failed"` and the
stages that did finish, so a half-populated results directory says how far it
got and why it stopped.

## Tests

```bash
pytest tests/          # 197 tests, CPU only, no downloads
```

The suite runs on tiny randomly initialized networks and synthetic HDF5 crops,
and checks properties rather than stored numbers: that the optimizer descends
its loss, that no pixel moves further than `iterations * step_size`, that
batched and single-image runs agree, that `L_+` lowers the similarity while
`L_-` raises it, that deletion and insertion masks are complementary, that
target layers resolve on all five architectures, and that a map written by the
run script is read back correctly by the evaluation. The supplemental paths are
covered too: that a larger step size moves the image further, that a wider blur
spreads the attribution, that the sweep varies one knob at a time, that the
timer reports elapsed seconds, and that a comparison sheet built from files on
disk leaves a blank cell where a method produced no map, and that a crashed
run still records the stages it completed.

The two equivalence files are the strongest checks in the suite. They
transcribe the originals literally, loops and all, and require the rewritten
code to match numerically:

| file | covers | transcribed from |
|---|---|---|
| `test_fame_equivalence.py` | FAME on both tasks, CAM on both tasks | `code_fame.py`, `utils.py`, `code_cam.py`, `cam.py`, `targets.py` |
| `test_baseline_equivalence.py` | CorrRISE, FGGB | `Corrise.py`, `code_fggb.py` |
| `test_evaluation_equivalence.py` | IoU, ROAD, deletion/insertion, AUC, EER | `eval_iou.py`, `eval_road_delete.py`, `eval_delete_fame.py`, `eval_insert_fame.py`, `eval_auc.py`, the scores/EER script |

Everything agrees to floating-point exactness, and each test fails if the
corresponding fix is reverted -- the grayscale ordering, the prescale, the
cosine denominators, the FGGB gradient normalization, the threshold scaling,
the CorrRISE rank transform, the percentile masks and the ROAD masking order
are all pinned.

Where the original and the corrected behavior differ, both are reachable, so a
rewritten pipeline can be checked against previously reported numbers:
`--original-boxes`, `--road-imputation`, `classification_loss(absolute=True)`,
`fggb(per_dimension=True)`, `corrise(rank_based=False)` and
`cosine_similarity(detach_norms=True)`.

What it does **not** cover: the suite never loads pretrained weights or real
datasets, so it cannot tell you whether the numbers in Tab. 1 and Tab. 2
reproduce. Run one protocol end to end against your checkpoints before trusting
the reorganized pipeline.

## Notes

- `scores_eer.csv` stores the threshold formatted to four decimals, so the
  value the original pipeline operated on is rounded. `eval_verification.py`
  recomputes it at full precision by default; pass `--threshold` with the
  stored number to reproduce previously reported accuracies exactly.
- Early stopping is on for verification (`--epsilon 0.01`, as in `utils.lots`)
  and off for classification. Watch it on impostor pairs: `L_+ = s` can already
  be below 0.01 at iteration zero, in which case LOTS returns the image
  untouched and the map comes out empty. Pass `--epsilon 0` to disable it.
- `FameConfig(normalization="max")` follows the paper's max-normalization;
  the default `"minmax"` reproduces the original code, which subtracts the
  minimum first. The two differ noticeably for heavily blurred maps.
- `cosine_similarity(detach_norms=False)` is the default, matching
  `Cosine_Loss` in `utils.py`, where `compute_norm` is an ordinary
  differentiable operation. `True` freezes the denominators, which is what the
  CAM target does since it precomputes the norm product.
- The CAM baselines are checked against `pytorch_grad_cam`. The original ran a
  vendored rewrite of it that works on tensors rather than numpy; the only step
  where that could diverge is the upsampling to input resolution, and cv2 and
  `F.interpolate` agree there to float32 epsilon with identical percentile
  masks, so the comparison carries over.
- CorrRISE and the CAM baselines need `grad-cam`; everything in `fame/` other
  than `baselines/cam.py` runs without it.
