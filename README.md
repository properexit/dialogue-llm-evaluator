# Dialogue-Based Multilingual LLM Evaluation

This repository contains an interactive evaluation framework for analyzing chat-optimized Large Language Models (LLMs) using constraint-driven, dialogue-based games.

The project introduces a cooperative sentence-building game and uses it as a benchmark to study fluency, rule-following, creativity, and multilingual robustness. Experiments compare multiple LLMs against a human baseline in English and Urdu.

This work was developed as part of a Master’s project in Cognitive Systems and is accompanied by a detailed evaluation report in the `report/` directory.

---

## Motivation

Most LLM benchmarks rely on static, single-turn tasks. In contrast, real-world language use is interactive, incremental, constrained, and multilingual.

Dialogue-based games expose failure modes that static benchmarks often miss, including:
- rule violations
- over-generation
- thematic drift
- degraded performance in low-resource languages

This project addresses these gaps using game-based evaluation built on top of the ClemBench framework.

---

## Game Design Overview

The evaluation task is a cooperative sentence-building game inspired by word-guessing games such as Taboo and Codenames.

### Roles
- Helper: Extends a shared sentence by up to three words per turn without revealing the target word.
- Seeker: Guesses one word per turn to identify the hidden target.

### Core Constraints
- Fixed number of rounds (7)
- Strict word-count limits per turn
- No direct use or morphological variants of the target word
- Sequential sentence construction without reordering

A centralized GameMaster enforces all rules and terminates invalid games.

---

## Multilingual Evaluation

The game is language-agnostic by design. Linguistic content is fully separated from game logic.

Implemented languages:
- English (high-resource)
- Urdu (low-resource, right-to-left script)

Urdu word pairs were generated using a noun corpus and semantic similarity filtering. The same game logic and evaluation metrics are applied across languages.

---

## Experimental Setup

- Instances: Noun pairs categorized by semantic similarity
  - Easy: high similarity (e.g., Robot → Automation)
  - Hard: low similarity (e.g., Time → Text)
- Models evaluated:
  - GPT-3.5 and GPT-4 variants
  - Open-source instruction-tuned models (e.g., LLaMA, Gemma)
- Baseline: Human players acting as Seekers
- Trials: Fixed number of randomly sampled instances per difficulty level

---

## Evaluation Metrics

### Automatic Metrics
- Constraint adherence
- Word-count violations
- Target word leakage
- Turn exhaustion
- Sentence coherence and grammaticality

### Manual Metrics
- Creativity
- Appropriateness
- Thematic consistency

This combined evaluation captures both objective correctness and subjective language quality.

---

## Key Findings

- Fluent output does not guarantee rule compliance
- Larger models show more stable constraint adherence
- Significant performance degradation occurs in Urdu
- Human players achieve perfect rule adherence and higher creativity
- Common failure modes include over-generation, thematic drift, and ignoring sequential constraints

These findings highlight current limitations in rule-following and multilingual generalization of LLMs.

---

## Repository Structure

```text
ROOT/
├── clembench/
│   └── gettothepoint/
│       ├── master.py              # GameMaster logic
│       ├── player.py              # Helper / Seeker behavior
│       ├── instancegenerator.py   # Instance handling
│       ├── clemgame.json          # ClemBench game definition
│       ├── in/                    # English & Urdu instances
│       └── resources/             # Prompts, configs, word lists
├── scripts/                       # Data & instance generation scripts
├── data/                          # Embeddings and corpora (not tracked)
├── results/                       # Experimental outputs
├── report/
│   └── Report_GetToThePoint.pdf
├── model_registry.json
└── README.md
```

---

## Running the Benchmark (ClemBench CLI)

### Activate environment
```bash
conda activate venv
```

### Inspect available components
```bash
clem list games
clem list backends
clem list models
```

You should see `gettothepoint` listed as an available game.

### Run the benchmark
```bash
clem run -g gettothepoint -m <model_name>
```

This command executes the dialogue-based game, logs all interactions, and computes per-game scores.

### Transcribe interactions
```bash
clem transcribe
```

Generates HTML transcripts for qualitative inspection.

### Compute scores and evaluation metrics
```bash
clem score
clem eval
```

---

## Reproducibility Note

Due to environment drift, dependency changes, and model API deprecations, re-running the experiments may require updates to ClemBench and model backends.

This repository is provided as a research artifact documenting benchmark design, evaluation methodology, and observed LLM behavior patterns.

---

## Contributions

- Custom dialogue-based evaluation game implemented in ClemBench
- Multilingual instance generation for English and Urdu
- Automatic and manual evaluation metrics
- Comparative analysis across multiple LLMs and a human baseline
- Error taxonomy for interactive LLM failures

---

## Framework Reference

ClemBench: https://github.com/clp-research/clembench

---

## License

MIT License

---

## Author

Uday Bhaskar Kale  
(MSc Cognitive Systems)
