# TruthfulQA Hallucination Mitigation via Text Normalization

A Colab notebook that tests whether surface-level text normalization reduces hallucination on TruthfulQA using Flan-T5-Large.

## What it does

Loads TruthfulQA (multiple_choice + generation configs, joined on question text), evaluates 800 questions under five conditions (raw baseline, text-only, entity-only, number/date-only, and combined), and reports accuracy, hallucination rate, average margin, and a per-category breakdown.

Scoring is log-probability based: each answer choice is scored by its cross-entropy under the prompt, and the highest-scoring choice wins.

## Run it

1. Open in Colab, set runtime to GPU (T4 is fine).
2. Run cells top to bottom. Cell 1 installs deps, Cell 6 loads the model (~3 GB), Cells 9–10 run evaluation.
3. Expect 30–60 minutes on T4.

## Tunable

`SAMPLE_SIZE` (default 800, max 817) and `MODEL_NAME` (default `google/flan-t5-large`). Seeds are fixed at 42 for reproducibility.
