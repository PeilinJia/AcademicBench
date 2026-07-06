# AcademicBench: Benchmarking Multimodal Logical Reasoning over Academic Diagrams

**AcademicBench** is a benchmark for evaluating multimodal large language models (MLLMs) on structured reasoning over academic framework diagrams.

Academic framework diagrams are widely used in research papers to describe system architectures, algorithmic workflows, module relations, and conceptual structures. However, existing multimodal benchmarks mainly focus on natural images, document understanding, or statistical charts, providing limited diagnosis of whether MLLMs can reason over structured academic diagrams or abstain when queried relations are absent.

> Status: Submitted to NLPCC 2026, under review.

---

## Overview

AcademicBench evaluates MLLMs across three levels of diagram understanding:

1. **Element Perception**  
   Identifying entities, modules, components, and visual elements in academic diagrams.

2. **Relation Cognition**  
   Understanding directed relations, edge labels, dependencies, and module interactions.

3. **Function Summarization**  
   Summarizing the overall function and high-level purpose of a framework diagram.

The benchmark further introduces **hidden abstention testing**, where unanswerable relation queries are mixed with answerable questions to evaluate whether models over-answer when queried relations are absent.

---

## Benchmark Scale

- **300** academic framework diagrams
- Approximately **1,910** automatically generated questions
- **9** question types across three evaluation levels
- **9** mainstream MLLMs evaluated
- Graph-based annotation with typed entities, directed relational edges, edge labels, and global functional summaries

---

## Annotation Schema

Each diagram is annotated as a directed graph:

```json
{
  "entities": [
    {
      "id": "E1",
      "type": "module",
      "name": "Input Encoder"
    },
    {
      "id": "E2",
      "type": "module",
      "name": "Reasoning Module"
    }
  ],
  "relations": [
    {
      "source": "E1",
      "target": "E2",
      "label": "feature representation"
    }
  ],
  "function_summary": "The framework encodes input data and performs reasoning through modular interaction."
}
