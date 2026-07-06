import os
import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set


VALID_ENTITY_TYPES = {"Node", "Container"}
VALID_PREDICATES = {"flows_to", "contains", "connects_to"}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


class DiagramGraph:
    def __init__(self, annotation: dict):
        self.raw = annotation

        self.image_id = annotation.get("image_id", "")
        self.paper_id = annotation.get("paper_id", "")

        self.summary_text = ""
        if isinstance(annotation.get("summary"), dict):
            self.summary_text = str(annotation["summary"].get("text", "")).strip()
        elif isinstance(annotation.get("summary"), str):
            self.summary_text = str(annotation.get("summary", "")).strip()

        # 兼容新旧读取方式
        self.summary = {"text": self.summary_text}

        self.entities = []
        self.relations = []

#这四个字典就是整个图的核心数据结构。id2entity 是按ID查实体,text2ids 是按文本查实体（用来检测重名）,
# out_edges 和 in_edges 分别存每个节点的出边和入边。后面所有的查询操作——找后继、找容器、找路径——都是在这四个字典上做的。
        self.id2entity: Dict[str, dict] = {}
        self.text2ids: Dict[str, List[str]] = defaultdict(list)

        self.out_edges: Dict[str, List[dict]] = defaultdict(list)
        self.in_edges: Dict[str, List[dict]] = defaultdict(list)

        self._load_entities(annotation.get("entities", []))
        self._load_relations(annotation.get("relations", []))

    # -------------------------
    # Load / normalize
    # -------------------------
    def _load_entities(self, entities: List[dict]):
        for idx, ent in enumerate(entities):
            ent_id = str(ent.get("id", f"e{idx}")).strip()
            text = str(ent.get("text", "")).strip()
            ent_type = str(ent.get("type", "Node")).strip()

            if not ent_id:
                ent_id = f"e{idx}"
            if ent_type not in VALID_ENTITY_TYPES:
                ent_type = "Node"

            norm_ent = {
                "id": ent_id,
                "text": text,
                "type": ent_type
            }
            self.entities.append(norm_ent)
            self.id2entity[ent_id] = norm_ent
            self.text2ids[normalize_text(text)].append(ent_id)

    def _load_relations(self, relations):
        EMPTY_LABEL_TOKENS = {"", "none", "null", "n/a", "na", "-", "--"}
        for rel in relations:
            subj = str(rel.get("subject_id", "")).strip()
            obj = str(rel.get("object_id", "")).strip()
            pred = str(rel.get("predicate", "flows_to")).strip()
            edge_text = str(rel.get("edge_text", "")).strip()
            # Normalize placeholder tokens to empty string.
            if edge_text.lower() in EMPTY_LABEL_TOKENS:
                edge_text = ""
            is_conflict = bool(rel.get("is_conflict", False))

            if subj not in self.id2entity or obj not in self.id2entity:
                continue
            if subj == obj:
                continue
            if pred not in VALID_PREDICATES:
                continue

            norm_rel = {
                "subject_id": subj,
                "predicate": pred,
                "object_id": obj,
                "edge_text": edge_text,
                "is_conflict": is_conflict,
            }
            self.relations.append(norm_rel)
            self.out_edges[subj].append(norm_rel)
            self.in_edges[obj].append(norm_rel)

    # -------------------------
    # Basic entity helpers
    # -------------------------
    def get_entity(self, entity_id: str) -> Optional[dict]:
        return self.id2entity.get(entity_id)

    def get_entity_text(self, entity_id: str) -> str:
        ent = self.get_entity(entity_id)
        return ent["text"] if ent else ""

    def get_entity_type(self, entity_id: str) -> str:
        ent = self.get_entity(entity_id)
        return ent["type"] if ent else ""

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self.id2entity

    def find_ids_by_text(self, text: str, exact_normalized: bool = True) -> List[str]:
        if exact_normalized:
            return self.text2ids.get(normalize_text(text), [])
        hits = []
        q = normalize_text(text)
        for ent in self.entities:
            if q in normalize_text(ent["text"]):
                hits.append(ent["id"])
        return hits

    def all_entity_ids(self) -> List[str]:
        return [e["id"] for e in self.entities]

    def all_entity_texts(self) -> List[str]:
        return [e["text"] for e in self.entities]

    def all_entities(self) -> List[dict]:
        return list(self.entities)

    # -------------------------
    # Duplicate / uniqueness helpers
    # -------------------------
    def get_text_count(self, text: str) -> int:
        return len(self.text2ids.get(normalize_text(text), []))

    def has_duplicate_text(self, text: str) -> bool:
        return self.get_text_count(text) > 1

    def is_entity_text_duplicated(self, entity_id: str) -> bool:
        txt = self.get_entity_text(entity_id)
        if not txt:
            return False
        return self.has_duplicate_text(txt)

    def get_duplicate_text_entity_ids(self, text: str) -> List[str]:
        return list(self.text2ids.get(normalize_text(text), []))

    # -------------------------
    # Relation helpers
    # -------------------------
    def all_relations(self) -> List[dict]:
        return list(self.relations)

    def get_relations(
        self,
        predicate: Optional[str] = None,
        include_conflict: bool = True
    ) -> List[dict]:
        rels = self.relations
        if predicate is not None:
            rels = [r for r in rels if r["predicate"] == predicate]
        if not include_conflict:
            rels = [r for r in rels if not r["is_conflict"]]
        return rels

    def get_outgoing(
        self,
        entity_id: str,
        predicate: Optional[str] = None,
        include_conflict: bool = True
    ) -> List[dict]:
        rels = self.out_edges.get(entity_id, [])
        if predicate is not None:
            rels = [r for r in rels if r["predicate"] == predicate]
        if not include_conflict:
            rels = [r for r in rels if not r["is_conflict"]]
        return rels

    def get_incoming(
        self,
        entity_id: str,
        predicate: Optional[str] = None,
        include_conflict: bool = True
    ) -> List[dict]:
        rels = self.in_edges.get(entity_id, [])
        if predicate is not None:
            rels = [r for r in rels if r["predicate"] == predicate]
        if not include_conflict:
            rels = [r for r in rels if not r["is_conflict"]]
        return rels

    def edge_exists(
        self,
        subject_id: str,
        predicate: Optional[str],
        object_id: str,
        include_conflict: bool = True
    ) -> bool:
        for r in self.get_outgoing(subject_id, predicate=predicate, include_conflict=include_conflict):
            if r["object_id"] == object_id:
                return True
        return False

    # -------------------------
    # Flow helpers
    # -------------------------
    def get_successors(
        self,
        entity_id: str,
        flow_only: bool = True,
        include_conflict: bool = False
    ) -> List[str]:
        predicate = "flows_to" if flow_only else None
        rels = self.get_outgoing(entity_id, predicate=predicate, include_conflict=include_conflict)
        return [r["object_id"] for r in rels]

    def get_predecessors(
        self,
        entity_id: str,
        flow_only: bool = True,
        include_conflict: bool = False
    ) -> List[str]:
        predicate = "flows_to" if flow_only else None
        rels = self.get_incoming(entity_id, predicate=predicate, include_conflict=include_conflict)
        return [r["subject_id"] for r in rels]

    def get_unique_successor(self, entity_id: str, include_conflict: bool = False) -> Optional[str]:
        succs = self.get_successors(entity_id, flow_only=True, include_conflict=include_conflict)
        return succs[0] if len(succs) == 1 else None

    def get_unique_predecessor(self, entity_id: str, include_conflict: bool = False) -> Optional[str]:
        preds = self.get_predecessors(entity_id, flow_only=True, include_conflict=include_conflict)
        return preds[0] if len(preds) == 1 else None

    def get_successor_pairs(self, include_conflict: bool = False) -> List[Tuple[str, str]]:
        return [
            (rel["subject_id"], rel["object_id"])
            for rel in self.get_relations(predicate="flows_to", include_conflict=include_conflict)
        ]

    # -------------------------
    # Containment helpers
    # -------------------------
    def get_contained_children(self, container_id: str, include_conflict: bool = False) -> List[str]:
        rels = self.get_outgoing(container_id, predicate="contains", include_conflict=include_conflict)
        return [r["object_id"] for r in rels]

    def get_parent_containers(self, child_id: str, include_conflict: bool = False) -> List[str]:
        rels = self.get_incoming(child_id, predicate="contains", include_conflict=include_conflict)
        return [r["subject_id"] for r in rels]

    def get_unique_parent_container(self, child_id: str, include_conflict: bool = False) -> Optional[str]:
        parents = self.get_parent_containers(child_id, include_conflict=include_conflict)
        return parents[0] if len(parents) == 1 else None

    def get_containment_pairs(self, include_conflict: bool = False) -> List[Tuple[str, str]]:
        return [
            (rel["subject_id"], rel["object_id"])
            for rel in self.get_relations(predicate="contains", include_conflict=include_conflict)
        ]

    # -------------------------
    # Edge label
    # -------------------------
    def get_edges_with_text(
        self,
        predicate: Optional[str] = None,
        include_conflict: bool = False
    ) -> List[dict]:
        rels = self.get_relations(predicate=predicate, include_conflict=include_conflict)
        return [r for r in rels if str(r.get("edge_text", "")).strip()]

    def get_edge_text(self, subject_id: str, object_id: str, predicate: Optional[str] = None) -> Optional[str]:
        for rel in self.get_outgoing(subject_id, predicate=predicate, include_conflict=True):
            if rel["object_id"] == object_id:
                txt = str(rel.get("edge_text", "")).strip()
                return txt if txt else None
        return None

    # -------------------------
    # Connect relations
    # -------------------------
    def get_connected_neighbors(self, entity_id: str, include_conflict: bool = False) -> List[str]:
        rels = self.get_outgoing(entity_id, predicate="connects_to", include_conflict=include_conflict)
        return [r["object_id"] for r in rels]

    def get_connect_pairs(self, include_conflict: bool = False) -> List[Tuple[str, str]]:
        return [
            (rel["subject_id"], rel["object_id"])
            for rel in self.get_relations(predicate="connects_to", include_conflict=include_conflict)
        ]

    # -------------------------
    # Conflict
    # -------------------------
    def get_conflict_relations(self) -> List[dict]:
        return [r for r in self.relations if r["is_conflict"]]

    def get_non_conflict_relations(self) -> List[dict]:
        return [r for r in self.relations if not r["is_conflict"]]

    # -------------------------
    # Multi-hop path
    # -------------------------
    #这是一个BFS搜索,用来找两个节点之间的路径，用于多跳问题生成
    def find_path(
        self,
        source_id: str,
        target_id: str,
        predicates: Optional[Set[str]] = None,
        include_conflict: bool = False,
        max_hops: int = 4
    ) -> Optional[List[str]]:
        if source_id not in self.id2entity or target_id not in self.id2entity:
            return None
        if source_id == target_id:
            return [source_id]

        queue = deque([[source_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            if len(path) - 1 >= max_hops:
                continue

            for rel in self.get_outgoing(current, predicate=None, include_conflict=include_conflict):
                if predicates is not None and rel["predicate"] not in predicates:
                    continue

                nxt = rel["object_id"]
                if nxt in path:
                    continue

                new_path = path + [nxt]
                if nxt == target_id:
                    return new_path

                queue.append(new_path)

        return None

    def has_path(
        self,
        source_id: str,
        target_id: str,
        predicates: Optional[Set[str]] = None,
        include_conflict: bool = False,
        max_hops: int = 4
    ) -> bool:
        return self.find_path(
            source_id=source_id,
            target_id=target_id,
            predicates=predicates,
            include_conflict=include_conflict,
            max_hops=max_hops
        ) is not None

    def sample_multihop_pair(
        self,
        min_hops: int = 2,
        max_hops: int = 4,
        predicates: Optional[Set[str]] = None,
        include_conflict: bool = False,
        rng: Optional[random.Random] = None
    ) -> Optional[Tuple[str, str, List[str]]]:
        rng = rng or random
        entity_ids = self.all_entity_ids()
        candidates = []

        for src in entity_ids:
            src_text = self.get_entity_text(src).strip()
            if not src_text:
                continue

            for tgt in entity_ids:
                if src == tgt:
                    continue

                tgt_text = self.get_entity_text(tgt).strip()
                if not tgt_text:
                    continue

                path = self.find_path(
                    source_id=src,
                    target_id=tgt,
                    predicates=predicates,
                    include_conflict=include_conflict,
                    max_hops=max_hops
                )
                if path is None:
                    continue

                hops = len(path) - 1
                if not (min_hops <= hops <= max_hops):
                    continue

                path_texts = [self.get_entity_text(x).strip() for x in path]
                if any(not t for t in path_texts):
                    continue

                candidates.append((src, tgt, path))

        if not candidates:
            return None
        return rng.choice(candidates)

    # -------------------------
    # Top-level / simple structure heuristics
    # -------------------------
    def get_top_level_entities(self) -> List[str]:
        child_ids = set()
        for rel in self.get_relations(predicate="contains", include_conflict=False):
            child_ids.add(rel["object_id"])
        return [eid for eid in self.all_entity_ids() if eid not in child_ids]

    def get_root_flow_entities(self) -> List[str]:
        top_level = set(self.get_top_level_entities())
        roots = []
        for eid in top_level:
            preds = self.get_predecessors(eid, flow_only=True, include_conflict=False)
            preds = [p for p in preds if p in top_level]
            if len(preds) == 0:
                roots.append(eid)
        return roots

    def linearize_main_flow(self, max_steps: int = 20) -> List[str]:
        roots = self.get_root_flow_entities()
        if not roots:
            return []

        start = roots[0]
        chain = [start]
        visited = {start}
        current = start
        top_level_set = set(self.get_top_level_entities())

        for _ in range(max_steps - 1):
            succs = self.get_successors(current, flow_only=True, include_conflict=False)
            succs = [s for s in succs if s in top_level_set]

            if len(succs) != 1:
                break
            nxt = succs[0]
            if nxt in visited:
                break

            chain.append(nxt)
            visited.add(nxt)
            current = nxt

        return chain

    # -------------------------
    # Sampling helpers for QA generation
    # -------------------------
    def sample_presence_entity(self, rng: Optional[random.Random] = None) -> Optional[dict]:
        rng = rng or random
        candidates = [e for e in self.entities if str(e.get("text", "")).strip()]
        if not candidates:
            return None
        return rng.choice(candidates)

    def sample_absence_text(
        self,
        external_candidates: List[str],
        rng: Optional[random.Random] = None
    ) -> Optional[str]:
        rng = rng or random
        present = {normalize_text(e["text"]) for e in self.entities}
        pool = [x for x in external_candidates if normalize_text(x) not in present]
        if not pool:
            return None
        return rng.choice(pool)

    def sample_successor_question_target(self, rng: Optional[random.Random] = None) -> Optional[Tuple[str, str]]:
        rng = rng or random
        pairs = []
        for eid in self.all_entity_ids():
            if not self.get_entity_text(eid).strip():
                continue
            succ = self.get_unique_successor(eid, include_conflict=False)
            if succ is None:
                continue
            if not self.get_entity_text(succ).strip():
                continue
            pairs.append((eid, succ))
        if not pairs:
            return None
        return rng.choice(pairs)

    def sample_containment_target(self, rng: Optional[random.Random] = None) -> Optional[Tuple[str, str]]:
        rng = rng or random
        pairs = []

        # 只选择“唯一父容器”的 child，避免 containment 单选题潜在多答案
        for child_id in self.all_entity_ids():
            if not self.get_entity_text(child_id).strip():
                continue

            parent = self.get_unique_parent_container(child_id, include_conflict=False)
            if parent is None:
                continue

            if not self.get_entity_text(parent).strip():
                continue

            pairs.append((parent, child_id))

        if not pairs:
            return None
        return rng.choice(pairs)

    def sample_edge_label_target(self, rng: Optional[random.Random] = None) -> Optional[dict]:
        rng = rng or random
        candidates = self.get_edges_with_text(predicate=None, include_conflict=False)
        if not candidates:
            return None
        return rng.choice(candidates)

    def sample_conflict_target(self, rng: Optional[random.Random] = None) -> Optional[dict]:
        rng = rng or random
        conflicts = self.get_conflict_relations()
        if not conflicts:
            return None
        return rng.choice(conflicts)

    def sample_negative_relation(
        self,
        predicate: str,
        rng: Optional[random.Random] = None,
        max_trials: int = 100
    ) -> Optional[Tuple[str, str]]:
        rng = rng or random
        ids = [x for x in self.all_entity_ids() if self.get_entity_text(x).strip()]
        if len(ids) < 2:
            return None

        for _ in range(max_trials):
            a, b = rng.sample(ids, 2)
            if not self.edge_exists(a, predicate, b, include_conflict=True):
                return (a, b)
        return None


def load_annotation_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_diagram_graph(json_path: str) -> DiagramGraph:
    data = load_annotation_json(json_path)
    return DiagramGraph(data)


def load_graphs_from_dir(annotation_dir: str) -> List[DiagramGraph]:
    graphs = []
    if not os.path.exists(annotation_dir):
        return graphs

    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(annotation_dir, fname)
        try:
            graphs.append(load_diagram_graph(path))
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
    return graphs


# -------------------------
# Optional debug demo
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to one annotation json")
    args = parser.parse_args()

    graph = load_diagram_graph(args.json)

    print("=" * 60)
    print("image_id:", graph.image_id)
    print("paper_id:", graph.paper_id)
    print("summary:", graph.summary_text)
    print("-" * 60)

    print("Entities:")
    for e in graph.all_entities():
        print(e)

    print("-" * 60)
    print("Relations:")
    for r in graph.all_relations():
        print(r)

    print("-" * 60)
    print("Top-level entities:")
    print([(eid, graph.get_entity_text(eid)) for eid in graph.get_top_level_entities()])

    print("-" * 60)
    print("Main flow chain:")
    chain = graph.linearize_main_flow()
    print(chain)
    print([graph.get_entity_text(x) for x in chain])

    print("-" * 60)
    mh = graph.sample_multihop_pair(min_hops=2, max_hops=4, predicates={"flows_to"}, include_conflict=False)
    print("Sample multihop pair:", mh)
    if mh:
        src, tgt, path = mh
        print("Path texts:", [graph.get_entity_text(x) for x in path])

    print("-" * 60)
    print("Conflict relations:")
    for r in graph.get_conflict_relations():
        print(r)
