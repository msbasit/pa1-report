# Assignment 1: Temporal Modeling

**Task 1:** Heterogeneous Sensors &nbsp;|&nbsp; **Task 2:** Autoformer Challenge

AI651 – Deep Learning for Space, Time and Graphs · Fall 2026  
**Muhammad Sami Ul Basit** · Student ID 25280025

Full report: [25280025_PA1.pdf](25280025_PA1.pdf) · Repository: <https://github.com/msbasit/pa1-report>

## Summary

In **Task 1**, routing each forecast by the period estimated from its context beats every neural
forecaster tested. Decomposition and delay mixing still improve the neural models substantially.

**Task 2** applies a small Autoformer encoder–decoder to an anonymized, heavy-tailed series. The
optional known-future covariates give the largest measured improvement. The selected 3-member
ensemble reaches a local validation RMSE of **60.173** with **8,403** trainable parameters and
**24** production epochs in total. No hidden-target or public leaderboard score is recorded.

| Task | Problem | Selected approach | Headline result |
|---|---|---|---|
| **Task 1** | Forecast vibration across heterogeneous sensors (96 → 48 samples) | Period-routed ridge | Test RMSE **0.161** (established) / **0.183** (held-out station) m/s² |
| **Task 2** | Forecast an anonymized series 168 steps ahead with an Autoformer | 3 × 8-epoch Autoformer ensemble (width 8) with known-future covariates | Validation RMSE **60.173**, against **94.919** for the best baseline |

---

## Repository layout

```
.
├── 25280025_PA1.pdf                  # combined report (Tasks 1 & 2)
├── requirements.txt
├── Question 1/                       # Task 1 — heterogeneous sensors
│   ├── Assignment1.ipynb             # experiment notebook (full preset)
│   ├── report.tex                    # LaTeX report
│   ├── harness/                      # data, models, diagnostics, controls, reporting
│   ├── checkpoints/                  # trained neural model weights (*.pt)
│   ├── extracted_images/             # figures used in the report
│   └── results/design/               # per-output tables (CSV + LaTeX)
└── Question 2 - Leaderboard/         # Task 2 — Autoformer leaderboard
    ├── Task-2.ipynb                  # end-to-end pipeline
    ├── Data/                         # train, test stub, optional external covariates
    └── artifacts/                    # predictions.txt, filled test CSV, submission_meta.json
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Requires Python 3.10+ and PyTorch ≥ 2.4. Task 1 was run on CPU. Task 2 was run on Colab (PyTorch
2.11.0+cu130, CUDA) and also runs on CPU.

---

## Task 1 — Forecasting Across Heterogeneous Sensors

A simulated production line with four vibration stations sampled at 20 Hz. Products A, B, and C
have nominal periods of 14–18, 21–27, and 28–36 samples. Stations 1–3 are **established**;
station 4 is **held out** from fitting. Each forecaster sees a 96-sample context from one station
and predicts the next 48 samples.

- **Splits (target intervals):** train `[0, 9600)`, validation `[9600, 12480)`, test `[12480, 15360)`
- **Run:** `PA1_PRESET=full`, data seed 0, model seed 0, CPU, 6,000 optimization steps per neural model
- **Metrics:** RMSE (m/s²) and MSE ((m/s²)²); lower is better

### Part 1 — Context-dependent forecasting (validation)

| Model | Est. RMSE | Held-out RMSE | Params | Fit time |
|---|---:|---:|---:|---:|
| Shared ridge | 0.767 | 0.876 | 4,608 | 0.002 s |
| Sensor-specific ridge | 0.768 | n/a | 13,824 | 0.003 s |
| **Period-routed ridge** | **0.166** | **0.173** | 59,904 | 0.006 s |
| Raw Attention | 0.431 | 0.612 | 368,368 | 760 s |

The results follow the hypothesized ranking (stated retrospectively, after results existed):
period-routed ridge < Raw Attention < shared ≈ sensor-specific ridge. Station identity alone
doesn't help, because one station runs at several periods (Station 1 runs at 16.135 samples under
Product A and 30.093 under Product C). Compared with shared ridge, routed ridge cuts RMSE by about
**78% / 80%** (established / held-out) and Attention by **44% / 30%**.

Attention's mixing weights do change with context: the mean/max absolute weight difference between
contexts is 0.0113 / 0.4250. That shows the weights depend on the input, not that the model has
recovered the physical period.

<p align="center">
  <img src="Question%201/extracted_images/01-cell-12-output-02.png" width="85%" alt="Forecasts for Product A and Product C on Station 1">
</p>

### Part 2 — Series decomposition

A centered moving average with replicated endpoints splits the input into a trend and a remainder:
$T_t=\frac{1}{m}\sum_{u=-q}^{q}x_{t+u}$ and $S = x - T$. The seasonal part goes to the encoder and
the trend to a linear head.

| Representation | Residual std | Recovery within 1 sample | Median error |
|---|---:|---:|---:|
| Raw | – | 68.67% | 0.525 |
| Width 17 | 0.578 | 98.00% | 0.157 |
| Width 25 | 1.012 | 98.92% | **0.122** |
| Width 33 | 1.372 | 99.33% | 0.142 |
| **Width 41 (selected)** | 1.572 | **99.85%** | 0.216 |

Width 41 (2.05 s) gives the best within-one-sample oracle recovery on training windows, while
width 25 has the lowest median error. The known synthetic period is used only to score this
diagnostic and is never an input to the model.

| Model | Est. RMSE | Held-out RMSE |
|---|---:|---:|
| Raw Attention | 0.431 | 0.612 |
| Attention + decomposition | **0.296** (−31%) | **0.430** (−30%) |

<p align="center">
  <img src="Question%201/extracted_images/03-cell-18-output-02.png" width="85%" alt="Moving-average decomposition widths">
</p>

### Part 3 — Delay-based mixing

Values are aggregated by delay with $z_t=\sum_{j=1}^{K} \alpha_j\, v_{(t-\tau_j)\bmod L}$, using
the 8–48 lag band, $K=\lceil 2\ln L\rceil = 10$, and a learned softmax temperature. Two checks pass:
the FFT scores match direct correlation ([4, −4, 4, −4]), and the aggregation example returns
[35, 15, 25, 25].

For Station 2 / Product A ($P = 16.135$), the residual recurrence peaks near 16 and 32 samples
($P$, $2P$) with correlations of 0.896 / 0.897, and again near 48 ($3P$).

<p align="center">
  <img src="Question%201/extracted_images/06-cell-31-output-02.png" width="85%" alt="Recurrence diagnostic">
</p>

### Part 4 — 2×2 design comparison and deployment

| Model | Est. RMSE | Held-out RMSE | Params |
|---|---:|---:|---:|
| Raw Attention | 0.431 | 0.612 | 368,368 |
| Attention + decomposition | 0.296 | 0.430 | 373,024 |
| Raw delay mixer | 0.221 | 0.340 | 368,370 |
| **Autoformer-inspired** (decomp + delay) | **0.199** | **0.239** | 373,026 |

Decomposition and delay mixing both help. On its own, delay mixing gives the larger aggregate
gain. Combining the two produces the best neural model overall, but the gains are
**subadditive**: the combined MSE reductions (0.146 / 0.317) are smaller than the summed
individual gains (0.236 / 0.447). The combined model is not best everywhere either. On
established Product A, raw delay mixing does better (0.211 vs. 0.226).

| Model | Est. step 1 | Est. step 48 | Held-out step 1 | Held-out step 48 |
|---|---:|---:|---:|---:|
| Period-routed ridge | 0.112 | **0.228** | 0.119 | **0.212** |
| Raw Attention | 0.144 | 0.528 | 0.176 | 0.761 |
| Raw delay mixer | 0.121 | 0.346 | 0.139 | 0.630 |
| Attention + decomposition | 0.149 | 0.399 | 0.147 | 0.589 |
| Autoformer-inspired | 0.135 | 0.237 | 0.147 | 0.299 |

<p align="center">
  <img src="Question%201/extracted_images/08-cell-36-output-02.png" width="85%" alt="RMSE across the forecast horizon">
</p>

**Deployment (chosen on validation, then scored once on test):**

| Population | Model | Val. RMSE | Test RMSE |
|---|---|---:|---:|
| Established | Period-routed ridge | 0.166 | **0.161** |
| Held-out | Period-routed ridge | 0.173 | **0.183** |

Period-routed ridge is chosen for both populations. It has the lowest validation error, 59,904
parameters, and the fastest fit among the competitive models. Test results were not used to
choose it.

> **Takeaway:** the right inductive bias matters more than model size here. A 60k-parameter routed
> ridge model that fits in milliseconds beats every 370k-parameter neural model on this synthetic
> benchmark. These are single-seed results.

Task 1 source: [Question 1/report.tex](Question%201/report.tex). Per-output tables:
[Question 1/results/design](Question%201/results/design).

---

## Task 2 — Leaderboard Challenge: Autoformer

The task is to forecast `value` for `time_idx = 43657 … 43824` (exactly **168 steps**) with an
Autoformer (Wu et al., 2021). Submissions are ranked by

$$\text{Score} = \text{RMSE} + \alpha \cdot P + \beta \cdot E$$

where $P$ is the number of trainable parameters and $E$ is the number of training epochs.

### Data and exploration

- The history has 43,656 observations. The ten optional covariates cover all 43,824 positions,
  including the forecast horizon. Units and dates are not given.
- The target is heavy-tailed: mean 98.17, sd 90.83, median 73, IQR 30–136, range 0–994, skewness 1.83.
- The periodogram gives a **24-step cycle**. The ACF at lags 1 / 24 / 168 is 0.966 / 0.401 / 0.028
  and first drops below 0.1 at lag 55, so linear persistence is weak over the full horizon.
- Fourier features use 24-, 168-, and 365.25×24-step cycles, each with two harmonics. The
  "day/week/year" names are assumptions, not recovered calendar dates.

### Model and preprocessing

| Component | Choice |
|---|---|
| Architecture | From-scratch Autoformer: 2 encoder / 1 decoder layers, 4 heads, progressive moving-average decomposition (width 25), FFT Auto-Correlation with top-k delay aggregation |
| Size | `d_model=8`, `d_ff=16`, dropout 0, context / warm-start / horizon = 168 / 168 / 168 |
| Target | Square root, standardized globally on the early fitting slice, with an affine inverse correction fitted on calibration blocks only (scale 0.9313, offset +17.506) |
| Features (48) | 10 external levels (D/E/F log-transformed), 6 differences, 12 trailing 6- and 24-step means, 8 regime × phase interactions, 12 Fourier terms. Past stamps feed the encoder and **known-future stamps feed the decoder** |
| Optimization | AdamW (lr 1e-3, wd 1e-4), batch 64, grad clip 1, 5% warm-up then cosine decay, stride 1 |

Differences from the reference Autoformer: delays are selected per example, lag zero is excluded,
aggregation uses FFT circular convolution, and positional encodings are fixed sinusoids.

### Chronological split

| Role | Origins | Stride | Target positions |
|---|---:|---:|---|
| Training | 38,617 | 1 | [168, 38952) |
| Affine calibration | 44 | 24 | [39288, 40488) |
| Reported validation | 132 | 24 | [40344, 43656) |
| Hidden forecast | 1 | – | [43656, 43824) |

### Baselines

| Model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Persistence | 93.091 | 133.895 | 90.31% |
| Seasonal naive (repeat prior 168-step block) | 94.273 | 133.381 | 96.40% |
| 24-step cycle climatology | 72.590 | **94.919** | 82.12% |

### Design screens (3 seeds, 4 epochs)

**Target transform** (width 32, 32,705 params):

| Transform | RMSE (mean ± sd) | MAE | sMAPE | Ensemble RMSE |
|---|---:|---:|---:|---:|
| Identity | 68.780 ± 1.079 | 46.804 | 59.78% | 66.717 |
| **Square root** | **64.701 ± 1.691** | 43.115 | 55.17% | 62.159 |
| Log1p | 65.184 ± 5.987 | 44.426 | 58.33% | 60.716 |

**Model width** (square-root target). The assumed score uses $\alpha = 2\times10^{-5}$ and $\beta = 0.05$:

| Width | Params | RMSE (mean ± sd) | Ensemble RMSE | Assumed score |
|---:|---:|---:|---:|---:|
| **8** | **2,801** | **63.347 ± 1.014** | **61.151** | **63.603** |
| 16 | 9,185 | 66.079 ± 1.108 | 64.384 | 66.463 |
| 32 | 32,705 | 64.701 ± 1.691 | 62.159 | 65.556 |

Width 8 has the lowest error and the fewest parameters, so it beats both larger models outright.

### Optional-external ablation

| Features | Params | RMSE (mean ± sd) | MAE | sMAPE | Ensemble RMSE |
|---|---:|---:|---:|---:|---:|
| Endogenous only | 2,225 | 99.810 ± 1.213 | 70.601 | 81.18% | 98.841 |
| External levels | 2,385 | 68.720 ± 1.320 | 47.832 | 59.39% | 67.152 |
| + differences | 2,481 | 66.462 ± 2.943 | 46.604 | 59.29% | 64.595 |
| **+ differences + context** | 2,801 | **63.347 ± 1.014** | **42.622** | **54.15%** | **61.151** |
| Cycle climatology | 0 | 94.919 | 72.590 | 82.12% | 94.919 |

The full external context lowers mean RMSE by **36.5%** compared with endogenous-only and by
**33.3%** compared with climatology. Most of the gain comes from the levels. The extra gains from
differences and context are small next to the seed spread.

### Epoch budget and ensemble cost

The assumed score is $S = \text{RMSE} + 2\cdot10^{-5}P + 0.05E$ (assumed penalties, not the official ones):

| Epochs/member | Members | Val. RMSE | Total P | Total E | Assumed S |
|---:|---:|---:|---:|---:|---:|
| **8** | **3** | **60.173** | 8,403 | 24 | **61.541** |
| 4 | 3 | 61.151 | 8,403 | 12 | 61.919 |
| 8 | 1 | 62.873 | 2,801 | 8 | 63.329 |
| 4 | 1 | 63.347 | 2,801 | 4 | 63.603 |
| 2 | 3 | 64.166 | 8,403 | 6 | 64.634 |
| 2 | 1 | 65.916 | 2,801 | 2 | 66.072 |

At 3 members, 8 epochs beats 4 epochs only if $\beta < (61.151 - 60.173)/12 \approx 0.0815$.

### Final submission

| Field | Value |
|---|---|
| Configuration | Square root, width 8, full externals, seeds 0/1/2, 8 epochs each |
| Production resources | $P = 3 \times 2{,}801 = $ **8,403**; $E = 3 \times 8 = $ **24** |
| Final refit | 43,321 origins (stride 1), about 701 s in total |
| Forecast range | min 17.83, mean 132.71, max 267.06 |
| Local metrics | Ensemble RMSE **60.173**; per-seed RMSE 62.873 ± 0.815, MAE 43.160, sMAPE 56.04% |
| Leaderboard result | Not recorded; official score unknown |

### Caveats

- **Calibration and validation overlap.** Their target ranges share 144 samples, so the reported
  numbers are not fully independent validation.
- **Overlapping windows.** The 132 rolling windows overlap. The seed SD measures initialization
  variance, not uncertainty across independent blocks.
- **Different averaging rules.** Validation averages predictions after inverse-transforming them;
  production averages before the inverse. The 60.173 figure therefore does not directly evaluate
  the final averaging rule.

### Artifacts

- [predictions.txt](Question%202%20-%20Leaderboard/artifacts/predictions.txt): the 168 comma-separated forecasts
- [student_test_filled.csv](Question%202%20-%20Leaderboard/artifacts/student_test_filled.csv): the test file with forecasts filled in
- [submission_meta.json](Question%202%20-%20Leaderboard/artifacts/submission_meta.json): $P$, $E$, full config, calibration, and validation metrics

---

## Conclusions and limitations

- **Task 1:** a linear model conditioned on the matched period beats extra model flexibility. The
  two neural upgrades still improve aggregate accuracy.
- **Task 2:** the permitted known-future covariates and a compact Autoformer give strong gains.
  External-driven spikes are still hard to predict. Any cost comparison has to count every
  ensemble member and every epoch.
- **Scope:** Task 1 is a single-seed synthetic validation plus one final test. Task 2 is
  three-seed model-selection validation with overlapping targets, assumed cost weights, and no
  public leaderboard result.

## References

1. Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.*
2. Vaswani, A. et al. (2017). *Attention Is All You Need.*
3. Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?*
4. Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice.*
