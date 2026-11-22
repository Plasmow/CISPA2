# CISPA2

Repository for the CISPA European AI Championship second edition on watermarking for Automateam2.

Each task has its own folder, with its own README (method, code pointers, results).

| Task | Folder | Description |
|---|---|---|
| Watermark localisation | [`watermark_localisation/`](watermark_localisation/) | Predict a 128×128 mask of the watermarked region of each edited image (PixelSeal, TrustMark, StegaStamp). Best public score 0.7324. |
| Unlearning | [`unlearning/`](unlearning/) | Make a ViT-B/16 classifier behave as if it had never seen 3,500 training images, by designed-target distillation with diffusion-generated anchors. Public score 0.400. |
| Model equality | [`model_equality/`](model_equality/) | For 33 endpoints, find which share the same base model family and which serve exactly the same model (MMD on responses, white-box and token-level features). |
