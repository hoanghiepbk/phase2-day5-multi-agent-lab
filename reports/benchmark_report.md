# Benchmark Report

- **Author:** Phạm Hữu Hoàng Hiệp
- **Student ID:** 2A202600415
- **Generated:** 2026-05-06 05:06:10 UTC

## 1. Per-run results

| Run | Query | Latency (s) | Cost (USD) | Quality | Citation cov. | Iters | Failure | Notes |
|---|---|---:|---:|---:|---:|---:|:---:|---|
| baseline | Research GraphRAG state-of-the-art and write a 5… | 23.39 | 0.0004 | 4.0 | 0.00 | 0 | no | iters=0, sources=0, errors=0, backend=openai |
| multi-agent | Research GraphRAG state-of-the-art and write a 5… | 50.74 | 0.0012 | 8.7 | 0.33 | 4 | no | iters=4, sources=5, errors=1, backend=openai |
| baseline | Compare single-agent and multi-agent workflows f… | 30.80 | 0.0003 | 4.0 | 0.00 | 0 | no | iters=0, sources=0, errors=0, backend=openai |
| multi-agent | Compare single-agent and multi-agent workflows f… | 17.47 | 0.0010 | 8.7 | 0.35 | 4 | no | iters=4, sources=5, errors=1, backend=openai |
| baseline | Summarize production guardrails for LLM agents | 5.26 | 0.0002 | 4.0 | 0.00 | 0 | no | iters=0, sources=0, errors=0, backend=openai |
| multi-agent | Summarize production guardrails for LLM agents | 20.00 | 0.0009 | 8.8 | 0.40 | 4 | no | iters=4, sources=5, errors=1, backend=openai |

## 2. Aggregate by run

| Run | Mean latency (s) | Mean cost (USD) | Mean quality | Mean citation cov. | Failures |
|---|---:|---:|---:|---:|---:|
| baseline | 19.81 | 0.0003 | 4.00 | 0.00 | 0/3 |
| multi-agent | 29.40 | 0.0010 | 8.73 | 0.36 | 0/3 |

## 3. Baseline vs Multi-agent

- Latency multiplier (multi / baseline): **1.48x**
- Cost delta: baseline $0.0003 -> multi $0.0010
- Quality delta: baseline 4.00 -> multi 8.73

## 4. Failures observed

No failures recorded across the benchmark set.

