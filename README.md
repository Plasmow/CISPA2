# CISPA2

Repository for the CISPA European AI Championship second edition on watermarking for Automateam2.

Each task has its own folder, with its own README (method, code pointers, results).

| Task | Folder | Description |
|---|---|---|
| Watermark localisation | [`watermark_localisation/`](watermark_localisation/) | Predict a 128×128 mask of the watermarked region of each edited image (PixelSeal, TrustMark, StegaStamp). Best public score 0.7324. |
| Unlearning | [`unlearning/`](unlearning/) | Make a ViT-B/16 classifier behave as if it had never seen 3,500 training images, by designed-target distillation with diffusion-generated anchors. Public score 0.400. |
| Model equality | [`model_equality/`](model_equality/) | For 33 endpoints, find which share the same base model family and which serve exactly the same model. Best public score 0.60. |

## Model equality: summary

For 33 endpoints (`ep-01` to `ep-33`), say which are built on the same base model (family) and which serve exactly the
same model (same checkpoint, quantisation and LoRA). A watermark, a system prompt, a persona or a change of decoding
does not change the model. Score = 0.1 × MCC "same family" + 0.9 × MCC "same model" (pairs within a family), with a
budget of 100 responses per endpoint.

**Final partition** (`submission_v20.csv`, public score 0.60):

| Family | Base model | Core (same model as the base) | Distinct models |
|---|---|---|---|
| A | Llama-3.2-1B-Instruct | 09 10 13 16 19 29 31 | 05, 11, 15 (fp8), 27 |
| B | Qwen2.5-1.5B-Instruct | 03 06 08 20 21 26 30 32 | 07, 22 (fp8), 28 |
| C | Qwen2.5-3B-Instruct | 01 04 12 14 17 23 24 25 33 | 02, 18 |

**Method in short**
1. **Families:** similarity of responses to the same prompts, crossed with responses generated locally by candidate
   models. The partition is 100 % stable (even vs odd prompts): three families of 11 endpoints.
2. **Disguises:** personas, system prompts, watermarks and temperature 1.0 leave detectable traces in the text, which
   proves the weights are the base model's.
3. **White-box tests:** every response is re-read by the base model in bf16 and in quantised versions (simulated fp8,
   int8, nf4): exact-score test, fp8 likelihood shift, hidden-prompt search, white-box RUT.
4. **Validation on known ground truth,** then classifiers: synthetic endpoints built from our own base models with known
   modifications, a simple classifier on 22 features per endpoint and a deep (Deep Sets) classifier on per-token features.
5. **Decision:** the sources are combined according to their measured reliability, and every change is checked against
   the public-score history.

**Limit:** a weak LoRA on the server, different from our adapters, stays undetectable with about 95 responses
(the case of `ep-18`, isolated through the public score only).

Full write-up, results table and file guide: [`model_equality/README.md`](model_equality/README.md) (in French).
