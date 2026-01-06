# Get to the Point: Multilingual LLM Evaluation via Dialogue-Based Games

This repository contains a ClemBench-based interactive evaluation framework for analyzing the behavior of chat-optimized Large Language Models (LLMs) in constraint-driven, dialogue-based tasks.

The project implements a cooperative sentence-building game ("Get to the Point") and uses it as a benchmark for fluency, constraint adherence, creativity, and multilingual robustness. Experiments compare multiple LLMs against a human baseline in English and Urdu.

This work was developed as part of a Master’s project in Cognitive Systems and is accompanied by a detailed evaluation report (included in /report).

---

## Motivation

Most LLM benchmarks rely on static, single-turn tasks. However, real-world language use is:
- interactive
- incremental
- constrained
- multilingual

Dialogue-based games expose failure modes that static benchmarks often miss, such as:
- rule violations
- over-generation
- thematic drift
- degraded performance in low-resource languages

This project addresses that gap using game-based evaluation built on top of the ClemBench framework.

---

## Game Design: Get to the Point

Inspired by Taboo and Codenames, the game challenges two players to collaboratively build a sentence under strict rules.

### Roles
- Helper: Extends a sentence (up to 3 words per turn) without revealing the target word.
- Seeker: Guesses one word per turn to identify the hidden target.

### Constraints
- Fixed number of rounds (7)
- Word-count limits per turn
- No direct use (or variants) of the target word
- Sequential sentence construction (no reordering)

A GameMaster enforces all rules and terminates invalid games.

---

## Multilingual Extension

The game is language-agnostic by design. Linguistic content is separated from game logic.

Implemented languages:
- English (high-resource)
- Urdu (low-resource, right-to-left script)

Urdu word pairs were generated using a noun corpus and similarity filtering.
The same game logic and evaluation metrics are used across languages.

---

## Experimental Setup

- Instances: Noun pairs categorized by semantic similarity
  - Easy: high similarity (e.g., Robot → Automation)
  - Hard: low similarity (e.g., Time → Text)
- Models tested:
  - GPT-3.5, GPT-4 variants
  - Open-source models (e.g., LLaMA, Gemma)
- Baseline: Human players acting as Seekers
- Trials: Fixed number of randomly sampled instances per difficulty level

---

## Evaluation Metrics

### Automatic
- Constraint adherence
- Word-count violations
- Target word leakage
- Turn exhaustion
- Sentence coherence and grammar

### Manual
- Creativity
- Appropriateness
- Thematic consistency

This two-level evaluation captures both objective correctness and subjective language quality.

---

## Key Findings

- Fluency does not imply correctness
- Larger models show better rule adherence
- Significant multilingual degradation in Urdu
- Humans outperform models in constraint adherence and creativity
- Common failure modes include over-generation, thematic drift, and ignoring sequential constraints

These results highlight limitations in rule-following and multilingual generalization of current LLMs.

---

## Repository Structure

ROOT/
├── clembench/
│   └── gettothepoint/
│       ├── master.py
│       ├── player.py
│       ├── instancegenerator.py
│       ├── clemgame.json
│       ├── in/
│       └── resources/
├── scripts/
├── data/
├── results/
├── report/
│   └── Report_GetToThePoint.pdf
├── model_registry.json
└── README.md

---

## Running the Benchmark (ClemBench CLI)

Activate environment:
conda activate venv

Inspect components:
clem list games
clem list backends
clem list models

Run benchmark:
clem run -g gettothepoint -m <model_name>

Transcribe interactions:
clem transcribe

Compute scores:
clem score
clem eval

---

## Reproducibility Note

Due to environment drift, dependency changes, and model API deprecations, re-running experiments may require updates to ClemBench and model backends.

The repository is provided as a research artifact documenting benchmark design, evaluation methodology, and observed LLM behavior patterns.

---

## Contributions

- Custom dialogue-based evaluation game in ClemBench
- Multilingual instance generation (English and Urdu)
- Automatic and manual evaluation metrics
- Comparative analysis across multiple LLMs and a human baseline
- Error taxonomy for interactive LLM failures

---

## Framework Reference

ClemBench:
https://github.com/clp-research/clembench

---

## License

MIT License

---

## Author

Uday Bhaskar Kale
MSc Cognitive Systems
