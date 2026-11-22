# CISPA2
Repository for the CISPA European AI Championship second edition on watermarking for Automateam2

# Watermark localisation: team automateam2

**Best submission: `submission_v19b_sam.pt`, public score 0.7324** (submission 4223). The expected CV on train was 0.7332.

Each image is watermarked by **PixelSeal, TrustMark or StegaStamp** on a sub-region and then edited (outpainting,
object removal, addition or transplant). The method and the 100/256-bit message are known, on the test set too.
We predict a 128×128 binary mask, and the metric is the per-image 2-class mIoU.

Two main ideas carry the score:
- **Use the official watermark decoder as a feature extractor.** It knows what this watermark looks like and can
  check the known message.
- **Fuse many diverse localisation maps, then correct the shapes.** Detected rectangles must decode the message;
  objects are snapped to SAM segments.

## Locations

| Alias | Path | Content |
|---|---|---|
| `W/` | `/p/project1/training2666/debbah2/wmloc_unet/` | Main code, maps, runs, submissions. Full write-up in `W/BILAN_ARCHI.md` (French), experiment log in `W/COORD.md` |
| `loc/` | `Hackathon_Watermark_Localisation_Task/loc/` (this repository) | Data helpers, rectangle search, SAM candidates and snapping, cached decoder outputs (`loc/cache/`) |
| data | `Hackathon_Watermark_Localisation_Task/release/{train,test}` | 510 train images (170 per method), 1020 test images |
| models | `Hackathon_Watermark_Localisation_Task/starter_kit/models` | Official decoders |

Python: `Hackathon_Watermark_Localisation_Task/.venv/bin/python` (torch, timm, scipy, onnxruntime).

## Key observations
- **Two regimes.**
  - Type **T** (`full_watermark`): about 90 % of the image is watermarked, and the task is to find the *holes*.
  - Type **F**: about 9 % is watermarked, as a rectangle from outpainting (F-rect) or an object (F-seg).
  - The global signed confidence of the decoder, `g = mean(logits × (2·msg − 1))`, separates T from F with an AUC
    of about 0.999. See `W/stack/v3.py:g_of` and `loc/common.py:signed_conf`.
- **Sliding-window or crop decoding carries almost no signal:** the decoder needs context.
  Learned maps built on the decoder's *features* work far better.
- **For F-rect images, cropping exactly the right box and resizing it the way the official wrapper does makes the
  decoder read the message back** (bit acc ≈ 1). The rectangle can therefore be *verified*, not just guessed.

## Pipeline (v19b) and code pointers

### 1. Source maps (128×128 logits; out-of-fold on train, 5 folds; test = mean of the fold models)

**1a. wm-U-Net: the strongest single source, about 0.61 alone.** Its encoder is a copy of the official decoder.
- Code: `W/stack/wm_unet.py`.
  - `WMUNet` (l.43): PixelSeal ConvNeXt at 256 px, TrustMark ResNet-50 at 245 px.
  - `StegaUNet` (l.84): an exact PyTorch port of the StegaStamp ONNX decoder, STN included, in `W/stack/stega_torch.py:StegaDecoder`.
    It predicts in the STN-warped frame and maps the result back to the image grid with the inverse affine.
- Bottleneck: the decoder's per-position message-agreement map plus `g`. Skips come from the decoder stages,
  plus a full-resolution high-pass stem.
- Training:
  1. Pre-training on synthetic edits (`SynthStream`, generator `W/synth/generator.py`).
  2. 5-fold fine-tuning on the real images.
- Loss: BCE + soft mIoU (`W/unet.py:miou_loss`).
- Outputs in `W/stack/maps/<tag>_{train,test}_<method>.npz`, tags `wmunet*`.

**1b. "Stack" U-Nets on generic ImageNet encoders: the encoder zoo, the last big gain.**
- Code: `W/train_unet.py`, model `W/unet.py:UNet128`.
- Inputs: RGB + high-pass ×20, plus a global vector (method one-hot, `g`, bit acc) injected at the bottleneck.
  The decoder also receives stacked auxiliary maps (head logit, per-position detector confidence, heatmap).
- 5-fold training, dihedral TTA (`predict`, l.131). The architectures are listed in `VARIANTS` (l.32).
- Encoders in v19b:
  - EfficientNet-B3, EfficientNetV2-S, ResNet-50, RegNetY-032;
  - **zoo added on the last day:** `zhrw18` (HRNet-W18, the best, about 0.61 alone), `zregy040`, `zrx50`, `zr2n50`, `zeffb4`.
- Job scripts: `W/run_zoo_{A,B}.sh`. The folds run in parallel, about 16 min on 2 nodes.
  `W/merge_zoo.py` builds `runs/<v>/{oof_all,test_mean}.npz`.

**1c. Heads on frozen decoder features.**
- `W/stack/head_maps.py`: per-position logistic regression.
- `W/stack/head_multi.py`: multi-layer hypercolumns.
- `W/stack/head_synth*.py`: convolutional heads trained on synthetic data.
- `W/stack/remark_maps.py`: StegaStamp re-marking residual.

### 2. Fusion, threshold rule, post-processing: `W/stack/v3.py`
- **Fusion.** Per-cell logistic regression on `[clip(logit_s, ±8), g/5, g/5·logit_s]`
  (`fuse_feats` l.94, `fit_fusion` l.99, `oof_fusion` l.108). The candidates are single sources, top-2, top-3 and all;
  the one with the best CV wins.
- **Threshold rule.** The rule is chosen per method by CV; `emiou_none` wins for all 3 methods.
  - `emiou_none` (`W/stack/rule_emiou.py:emiou_rule` l.74, `rule_preds_plus` l.108): for each image, keep the top-k
    cells, with k chosen to maximise the *expected* soft mIoU under the fused probabilities. It has no fitted parameter.
- **Post-processing by predicted type.** `POST_T` / `POST_F` (l.210-214) fill small holes or drop small blobs.
  The choice is made by CV (`post_scores`, `cv_select`).

### 3. Decoder-verified rectangles (F-rect)
- **Search.** A grid search plus greedy refinement of boxes, scored by decoding the crop resized like the official
  wrapper: StegaStamp 400 bicubic, PixelSeal 256 bilinear, TrustMark 245.
  - `loc/rect_search.py` (`coarse_boxes`, `refine`).
  - `loc/ss_local_search.py`: local search for StegaStamp around the predicted blob.
  - `W/stack/rect_seeded.py`: seeds taken from the fused map.
- **Use.** If the best box decodes the message with bit acc ≥ a threshold chosen by CV (0.85 / 0.85 / 0.9),
  the mask becomes that box: `v3.py:load_rect` l.42, option `--rect`.

### 4. SAM snapping (F-seg)
- **Candidates.** About 100 SAM ViT-L segments per image, precomputed in `loc/gen_candidates.py` → `loc/cache/sam_cands/`.
- **Rule.** For F images without a rectangle, take the segment that maximises soft IoU with the map × inside/outside
  contrast. Replace the mask only above a score threshold chosen by CV: `W/stack/snap_sam_local.py:snap` (l.71),
  a copy of `loc/snap_sam.py`.
- **Driver.** `W/stack/sam_stage.py <submission.pt>` replays the version from its `.json`, checks CV and test masks,
  snaps, and keeps `<stem>_sam.pt` only if the CV improves. It adds about +0.02 to +0.024 CV.

## Reproducing v19b
```bash
PY=/p/scratch/training2666/ayadi1/automateam2/Hackathon_Watermark_Localisation_Task/.venv/bin/python
cd /p/project1/training2666/debbah2/wmloc_unet/stack
E="headmulti remark headsyn headsyn2 wmunet wmunet0 wmunetF wmunetF0 wmunetH wmunetH0 wmunet2 wmunet20"
R="effb3_hp_glob effb3_hp_glob_stack rgb_hp_glob_synth8k loc_effv2s_hp_glob_stack loc_r50_hp_glob_stack loc_regy032_hp_glob_stack \
   zregy040_hp_glob_stack zrx50_hp_glob_stack zr2n50_hp_glob_stack zhrw18_hp_glob_stack zeffb4_hp_glob_stack"
WMLOC_NO_RECT3=1 $PY v3.py --rect --rules_plus --runs $R --extras $E --out ../submissions/submission_v19b.pt --baseline_cv 9   # CPU, ~3 min
$PY sam_stage.py ../submissions/submission_v19b.pt                                                                          # CPU, ~2 min
```
Upload: `from submission_template import submit; submit(path)`, run from `starter_kit/`.
Never run `submission_template.py` as a script: it uploads automatically.

## Score progression (public leaderboard)

| Version | Main addition | Public score |
|---|---|---|
| v1 | Logistic heads + T/F switch rule | 0.567 |
| v3c | Fusion + decoder-verified rectangles | 0.663 |
| v4_wmunet | wm-U-Net (decoder as encoder) | 0.687 |
| v7_sam | Seeded rectangles + SAM snapping | 0.709 |
| v9_sam | Expected-mIoU threshold rule | 0.713 |
| v12a_sam | Stack U-Nets with other encoders (EffV2-S, R50, RegNetY) | 0.719 |
| **v19b_sam** | **Encoder zoo (HRNet-W18, RegNetY-040, ResNeXt50, Res2Net50, EffB4)** | **0.7324** |

## What did not help (details in `W/BILAN_ARCHI.md` §8 and `W/COORD.md`)
- Sliding-window or crop decoding, gradient attribution, re-marking (except a little on StegaStamp).
- Shape families with learned selection, a learned SAM-segment choice, decoding each SAM segment.
- Test pseudo-labels, VAE reconstruction maps, position priors, smoothing the fused map.
- Paper-inspired modules, each tested with a paired control:
  - attention blocks in the wm-U-Net;
  - a WAM-style per-cell bit-decoding head (`W/stack/wam_ft.py`; the bit loss stays at chance with 136 images per fold);
  - MELB block statistics (`W/stack/melb_blocks.py`);
  - EditGuard's degradation prompts, not built: train and test show no degradation shift, since all are PNG with
    matched noise and blockiness statistics.
- **Lesson:** architecture *diversity* of the fused U-Nets helped most. Add-on modules on one network did not.
