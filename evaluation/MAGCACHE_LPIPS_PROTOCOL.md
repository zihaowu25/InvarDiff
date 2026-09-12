# Low-cost paired LPIPS protocol for cache tuning

## What LPIPS can and cannot measure

LPIPS is useful here because cache and reference outputs can be generated with
the same model, prompt, seed, scheduler, sampling steps, and geometry.  Under
that strict pairing, frame-wise LPIPS measures perceptual drift introduced by
the cache policy.  It is not, by itself, a video metric: it does not judge
motion smoothness, temporal consistency, dynamics, or text alignment.

For video, report all of the following:

1. Frame-wise LPIPS-Alex mean, p95, and maximum over every decoded frame.
2. The absolute difference between the reference and candidate sequences of
   adjacent-frame LPIPS values.
3. RGB temporal-delta L1: the error between candidate and reference frame
   differences.

The two temporal signals are inexpensive screening diagnostics.  Final
shortlisted settings should additionally be checked with a video-native metric
or human pairwise review.

## Two-stage tuning budget

### Stage A: screening

- Four content types: static subject, camera motion, articulated motion, and
  dense/high-frequency motion.
- One seed per prompt.
- Preserve the production denoising-step count and native resolution.
- For expensive video models, use 17 frames while screening.
- Prefer generating the uncached reference immediately before its candidate in
  the same loaded process and on the same GPU. A shared reference may be reused
  only after zero-hit controls demonstrate byte-identical outputs across the
  relevant processes and GPUs.
- Sweep one cache family at a time.  MagCache screening must disable the
  Finegrained layer cache.

Rank settings on the Pareto frontier of effective skip ratio (or isolated
latency) versus LPIPS mean and p95.  Reject settings with a sharp p95/max or
temporal-error increase even when the mean remains small.  Do not treat a
single universal LPIPS cutoff as model-independent.

### Stage B: confirmation

- Re-evaluate only the two best Stage-A settings.
- Use at least eight prompts and two seeds.
- Restore the intended production frame count.
- Measure latency one job at a time because concurrent CPU offloading and disk
  traffic distort wall-clock comparisons.
- Add a video-native temporal metric and blinded human A/B review.

## Reproducibility rules

- Compare decoded RGB outputs with identical frame count, dimensions, and FPS.
- Isolate two-stage calibration in an earlier process, then freshly reload the
  serialized Cache Book for the measured paired-generation process.
- Run a zero-hit acceptance control. Its paired reference and candidate must be
  byte-identical (LPIPS exactly 0); otherwise the measurement protocol is not
  eligible for preset selection.
- Keep prompt rewriting and super-resolution disabled unless they are part of
  both reference and candidate pipelines.
- Record the exact E/K/R values, model revision, seed, scheduler, steps,
  geometry, cache hit ratio, and output hashes.
- Reuse the same LPIPS backbone and preprocessing for the whole sweep.

Use `paired_lpips.py` to produce the machine-readable pair metrics.
