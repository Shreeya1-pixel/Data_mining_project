# Geopolitical News-Linked Association Rule Mining of Nifty Sectoral Co-Movements

**BITS Pilani, Dubai Campus — Data Mining (Design Project)**  
**Team:** Shreeya Gupta (Lead), Akshit Varshney, Asitha Gayatri Rudraraju, Yashvir Singh Phogat

This repository is the runnable research pipeline behind the IEEE-style project report. It builds a geopolitical news dataset for India from GDELT, turns trading days into baskets of news and sector-state items, mines lagged association rules with Apriori, tests them under a news-only block-permutation null with FDR control, and compares a CBA-style GeoRule classifier against classical baselines.

> Associations under a fixed price skeleton — not a trading system and not a claim of causality.

---

## 1. Problem

Indian sectoral indices (IT, Bank, Pharma, Auto, …) do not move independently. Many shocks that hit them are geopolitical (conflict, energy, shipping, sanctions) and show up in news before or alongside the tape. Most equity mining work either:

- mines **price-only** co-movement rules, or  
- scores **news sentiment** and stops at a coefficient / Sharpe ratio.

What is usually missing is a **readable rule** an analyst can inspect, for example:

`{ENERGY_EVT_L1} → {Auto_DOWN}`  (support, confidence, lift)

plus a significance test that does **not** destroy the strong cross-sector correlation that already exists in Indian prices.

---

## 2. Design objectives

| ID | Objective |
|----|-----------|
| O1 | Construct **GeoNews-IN**: session-aligned, event-typed geopolitical news items for India from GDELT 2.0 GKG |
| O2 | Encode twelve Nifty sectoral closes as daily **UP / DOWN** basket items (with lag-1 copies) |
| O3 | Mine **lagged news → price** association rules (Apriori) |
| O4 | Test lift under a **news-only block permutation** (prices fixed) + Benjamini–Hochberg / BY FDR |
| O5 | Score a **GeoRule** classifier vs Decision Tree, k-NN, Naive Bayes, SVM (+ majority & price-only controls) |

---

## 3. System architecture

```
GDELT GKG (15-min zips)          NSE sectoral closes
        │                                  │
        ▼                                  ▼
 Broad geo / India-or-econ           returns → UP/DOWN/FLAT
 gate (download)                           │
        │                                  │
        ▼                                  │
 T2 / T3 + keyword relevance               │
        │                                  │
        ▼                                  │
 IST 15:30 session map ────────────────────┤
        │                                  │
        ▼                                  ▼
 Decayed GTI / event items ──────► Daily baskets (+ _L1)
                                           │
                                           ▼
                              Apriori (supp / conf / lift)
                                           │
                                           ▼
                     News-column block permutation + BH/BY
                                           │
                                           ▼
                        Seed-stable rules → GeoRule vs baselines
```

Pipeline modules (`research/dm_pipeline/`):

| Module | Role |
|--------|------|
| `harvest_gkg.py` | Stream GKG zips; broad geo gate; write parquet |
| `news_filter.py` | T1–T4 tiers, dedup, theme tagging |
| `relevance.py` | Keyword relevance gate (labelled classifier optional later) |
| `session.py` | IST close mapping (15:30) |
| `gti.py` | Daily shocks, `Δ_t` span norm, exponential decay |
| `prices.py` | NSE closes → returns → states |
| `baskets.py` | Day baskets + one-hot matrix |
| `mining.py` | Apriori + lagged-news filter |
| `null_test.py` | News-column block permutation |
| `fdr.py` | Benjamini–Hochberg / Benjamini–Yekutieli |
| `classifier.py` | CBA-style GeoRule + sklearn baselines |
| `pipeline.py` / `cli.py` | End-to-end experiment runner |

---

## 4. Method (technical summary)

### 4.1 Data

- **News:** five harvested GKG months (Feb/Mar 2022, Oct 2023, Apr 2024, Sep 2026). Four of ninety-six 15-minute files **per day**. On-disk harvest ≈ **116,015** rows under a broad gate; after T3 + relevance ≈ **988** articles used in mining.
- **Prices:** twelve Nifty sectoral indices + Nifty 50 from NSE Indices (stored under `research/data/pilot/closes_niftyindices.csv`).

### 4.2 Items and mining

- Each trading day = one transaction.  
- Items: sector `UP`/`DOWN`, decayed event flags (`*_EVT`), `GTI_HIGH` / `GTI_RISING`, plus lag-1 (`_L1`) copies.  
- Predictive family: **only `_L1` antecedents**, day-*t* price consequents.  
- Defaults: support ≥ 0.08, confidence ≥ 0.55, lift ≥ 1.2.

### 4.3 Significance

- Null: shuffle **news columns only** in blocks of length `L = 10` (`B = 1,000` in the reported run). Prices stay fixed so sector co-movement is not destroyed.  
- BH at `q = 0.10`; BY reported as the stricter check (kept nothing at the same `q` in our run).  
- Three seeds → intersection of **141** rules used on the hold-out.

### 4.4 Classification

- GeoRule: first matching lagged rule by confidence / lift / support (CBA-style); else training majority.  
- Same lagged one-hot features for DT, k-NN, Bernoulli NB, linear SVM.  
- Targets scored: Bank, IT, Energy, Metal, Auto, Pharma.

---

## 5. Reported results (snapshot)

From `research/artifacts/` (seed 42 / reported config):

| Quantity | Value |
|----------|------:|
| Lagged news→price rules tested | 872 |
| BH survivors (example seed) | 228 |
| Seed intersection | 141 |
| Hold-out mean GeoRule accuracy | 0.484 |
| Majority baseline | 0.500 |
| Hold-out ROC-AUCs | ~chance |

**Takeaway:** the stack is auditable and the short-horizon ranking result is negative. Core rules often fire but do not track labels out of sample. Read outputs as associations.

See `research/artifacts/RESULTS.md` and `summary.json` for the full dump.

---

## 6. Repository layout

```
Data_mining_project/
├── README.md
├── requirements.txt
├── docs/figures/          # architecture / EDA / metrics figures
├── research/
│   ├── dm_pipeline/       # source
│   ├── data/pilot/        # GKG parquet samples + NSE closes
│   └── artifacts/         # rules, classifiers, GTI, summary
└── tests/research/        # unit tests (session, GTI, mining, null, FDR, clf)
```

---

## 7. Setup & run

**Python 3.11+** recommended.

```bash
cd Data_mining_project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
```

### Unit tests

```bash
pytest tests/research -v
```

### Full experiment

```bash
# lighter smoke (fewer permutations)
python -m research.dm_pipeline.cli run \
  --news-months-only \
  --n-perm 200 \
  --train-end 2024-04-30 \
  --holdout-start 2026-09-01 \
  --sectors Bank,IT,Energy,Metal,Auto,Pharma

# report-scale null (slow)
python -m research.dm_pipeline.cli run \
  --news-months-only \
  --n-perm 1000 \
  --train-end 2024-04-30 \
  --holdout-start 2026-09-01 \
  --sectors Bank,IT,Energy,Metal,Auto,Pharma
```

Artifacts write to `research/artifacts/`.

### Filter-tier audit on one parquet

```bash
python -m research.dm_pipeline.cli tiers \
  research/data/pilot/gkg_pilot_rows_202203.parquet
```

### Re-harvest another GKG month (optional)

```bash
python -m research.dm_pipeline.harvest_gkg 2023-10
```

---

## 8. Design notes / known limits

1. **Session month filter:** months that receive ≥1 post-close mapped article are kept **in full** (spillover months Apr 2022, Nov 2023, May 2024). That inflates session count beyond the five harvest calendars. A tighter cut would restrict to harvest-calendar months only.  
2. **Relevance** is a keyword gate in this submission; a labelled title classifier was started (`label_sample_300.csv`) but not finished.  
3. **Hold-out** is one short month (n = 21). Treat classifier metrics as exploratory.  
4. **BY FDR** kept no rules at `q = 0.10` in the reported run; BH is the operating filter.  
5. GDELT and NSE Indices remain the property of their providers; redistribution follows their terms.

---

## 9. Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11 |
| Data | pandas, pyarrow |
| Mining | mlxtend Apriori |
| Null / FDR | NumPy block shuffle, BH/BY |
| Baselines | scikit-learn |
| News source | GDELT 2.0 GKG |
| Prices | NSE Indices historical closes |

---

## 10. Citation / course context

Academic design project for **Data Mining**, BITS Pilani Dubai Campus (I Sem 2026–27).  
Topic framing: *News-linked association rule mining of Nifty sectoral co-movements* (not retail market-basket analysis; not price-level forecasting).

If you reuse the code, please keep the association-vs-trading disclaimer intact.
