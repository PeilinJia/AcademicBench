import os
import json
import random
from collections import Counter
from typing import List, Dict, Any

from src.logic_engine import load_graphs_from_dir
from src.prompt_generator import (
    PromptGenerator,
    build_absence_candidate_pool_from_graphs,
    build_edge_label_candidate_pool_from_graphs,
    save_samples_to_json,
    save_samples_to_jsonl,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANNOTATION_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
DEFAULT_OUTPUT_JSON = os.path.join(PROJECT_ROOT, "benchmark_dataset.json")
DEFAULT_SEED = 42

DEFAULT_QUESTION_TYPES = [
    "element_presence_mcq",
    "element_absence_mcq",
    "relation_successor_hidden_abstain_mcq",
    "relation_successor_multiselect",
    "relation_containment_hidden_abstain_mcq",
    "relation_containment_multiselect",
    "relation_edge_label_hidden_abstain_mcq",
    "relation_multihop_multiselect",
    "function_summary_open",
]


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def split_graphs(
    graphs,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    graphs = list(graphs)
    rng = random.Random(seed)
    rng.shuffle(graphs)

    n = len(graphs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:n_train + n_val + n_test]

    return train_graphs, val_graphs, test_graphs


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    level_counter = Counter()
    qtype_counter = Counter()
    answer_type_counter = Counter()
    image_counter = Counter()

    for s in samples:
        level_counter[s.get("level", "UNKNOWN")] += 1
        qtype_counter[s.get("question_type", "UNKNOWN")] += 1
        answer_type_counter[s.get("answer_type", "UNKNOWN")] += 1
        image_counter[s.get("image_id", "UNKNOWN")] += 1

    return {
        "num_samples": len(samples),
        "num_images": len(image_counter),
        "by_level": dict(level_counter),
        "by_question_type": dict(qtype_counter),
        "by_answer_type": dict(answer_type_counter),
    }


def print_summary(name: str, samples: List[Dict[str, Any]]):
    summary = summarize_samples(samples)
    print(f"\n===== {name} =====")
    print(f"Num samples: {summary['num_samples']}")
    print(f"Num images: {summary['num_images']}")

    print("By level:")
    for k, v in sorted(summary["by_level"].items()):
        print(f"  {k}: {v}")

    print("By question type:")
    for k, v in sorted(summary["by_question_type"].items()):
        print(f"  {k}: {v}")

    print("By answer type:")
    for k, v in sorted(summary["by_answer_type"].items()):
        print(f"  {k}: {v}")


def generate_samples_for_graphs(
    graphs,
    question_types: List[str],
    max_per_type: int,
    seed: int = 42,
    hidden_abstain_unanswerable_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    # 只在当前 split 内构建 candidate pool，避免 train/val/test 之间信息混杂
    absence_pool = build_absence_candidate_pool_from_graphs(graphs)
    edge_label_pool = build_edge_label_candidate_pool_from_graphs(graphs)

    generator = PromptGenerator(
        rng=rng,
        absence_candidate_texts=absence_pool,
        edge_label_candidate_texts=edge_label_pool,
        hidden_abstain_unanswerable_ratio=hidden_abstain_unanswerable_ratio,
    )

    all_samples = []
    for graph in graphs:
        samples = generator.generate_for_graph(
            graph=graph,
            enabled_question_types=question_types,
            max_per_type=max_per_type
        )
        all_samples.extend(samples)

    return all_samples


def save_manifest(
    output_dir: str,
    config: Dict[str, Any],
    split_stats: Dict[str, Dict[str, Any]]
):
    manifest = {
        "config": config,
        "statistics": split_stats
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest -> {manifest_path}")


def save_split_file(
    samples: List[Dict[str, Any]],
    output_path: str,
    use_jsonl: bool = False
):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    ensure_dir(output_dir)

    if use_jsonl:
        save_samples_to_jsonl(samples, output_path)
    else:
        save_samples_to_json(samples, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", type=str, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_per_type", type=int, default=1)

    parser.add_argument(
        "--question_types",
        type=str,
        default=",".join(DEFAULT_QUESTION_TYPES),
        help="Comma-separated question types."
    )

    parser.add_argument(
        "--hidden_abstain_unanswerable_ratio",
        type=float,
        default=0.5,
        help="Among hidden-abstention question types, probability of generating unanswerable instances."
    )

    parser.add_argument("--jsonl", action="store_true", help="Save as jsonl instead of json.")
    parser.add_argument("--split", action="store_true", help="Whether to split into train/val/test.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    args = parser.parse_args()

    question_types = [x.strip() for x in args.question_types.split(",") if x.strip()]

    graphs = load_graphs_from_dir(args.annotation_dir)
    if not graphs:
        print(f"No annotation JSON files found in: {args.annotation_dir}")
        return

    print(f"Loaded {len(graphs)} annotated graphs from {args.annotation_dir}")

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    ensure_dir(output_dir)

    config = {
        "annotation_dir": os.path.abspath(args.annotation_dir),
        "output": output_path,
        "seed": args.seed,
        "max_per_type": args.max_per_type,
        "question_types": question_types,
        "hidden_abstain_unanswerable_ratio": args.hidden_abstain_unanswerable_ratio,
        "jsonl": args.jsonl,
        "split": args.split,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    split_stats = {}

    if args.split:
        train_graphs, val_graphs, test_graphs = split_graphs(
            graphs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        split_to_graphs = {
            "train": train_graphs,
            "val": val_graphs,
            "test": test_graphs
        }

        ext = "jsonl" if args.jsonl else "json"

        for split_name, split_graphs_list in split_to_graphs.items():
            samples = generate_samples_for_graphs(
                graphs=split_graphs_list,
                question_types=question_types,
                max_per_type=args.max_per_type,
                seed=args.seed,
                hidden_abstain_unanswerable_ratio=args.hidden_abstain_unanswerable_ratio,
            )

            split_output_path = os.path.join(output_dir, f"{split_name}.{ext}")
            save_split_file(
                samples=samples,
                output_path=split_output_path,
                use_jsonl=args.jsonl
            )

            print(f"Saved {split_name} -> {split_output_path}")
            print_summary(split_name, samples)
            split_stats[split_name] = summarize_samples(samples)

    else:
        samples = generate_samples_for_graphs(
            graphs=graphs,
            question_types=question_types,
            max_per_type=args.max_per_type,
            seed=args.seed,
            hidden_abstain_unanswerable_ratio=args.hidden_abstain_unanswerable_ratio,
        )

        save_split_file(
            samples=samples,
            output_path=output_path,
            use_jsonl=args.jsonl
        )

        print(f"Saved full benchmark -> {output_path}")
        print_summary("full", samples)
        split_stats["full"] = summarize_samples(samples)

    save_manifest(output_dir, config, split_stats)


if __name__ == "__main__":
    main()
