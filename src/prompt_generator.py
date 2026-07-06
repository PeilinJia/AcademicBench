import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from question_safety_validator import QuestionSafetyValidator


from src.logic_engine import DiagramGraph

print("USING PROMPT_GENERATOR FILE:", __file__)


OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "from", "by", "with", "and", "or",
    "is", "are", "be", "this", "that", "these", "those", "as", "at", "into", "via",
    "module", "modules", "component", "components", "block", "blocks", "stage", "stages",
    "layer", "layers", "step", "steps", "unit", "units", "diagram", "figure", "system",
}


def normalize_ws(text: Any) -> str:
    return " ".join(str(text).strip().split())


def norm_key(text: Any) -> str:
    return normalize_ws(text).lower()


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = norm_key(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(normalize_ws(x))
    return out


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def safe_json_dump(data: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_samples_to_json(samples: List[Dict[str, Any]], output_path: str):
    safe_json_dump(samples, output_path)


def save_samples_to_jsonl(samples: List[Dict[str, Any]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def build_absence_candidate_pool_from_graphs(graphs: List[DiagramGraph]) -> List[str]:
    pool = []
    for g in graphs:
        for ent in g.all_entities():
            txt = normalize_ws(ent.get("text", ""))
            if txt:
                pool.append(txt)
    return unique_preserve_order(pool)


def build_edge_label_candidate_pool_from_graphs(graphs: List[DiagramGraph]) -> List[str]:
    pool = []
    for g in graphs:
        for rel in g.all_relations():
            txt = normalize_ws(rel.get("edge_text", ""))
            if txt:
                pool.append(txt)
    return unique_preserve_order(pool)


class PromptGenerator:
    def __init__(
        self,
        rng: Optional[random.Random] = None,
        absence_candidate_texts: Optional[List[str]] = None,
        edge_label_candidate_texts: Optional[List[str]] = None,
        hidden_abstain_unanswerable_ratio: float = 0.5,
        node_candidate_texts: Optional[List[str]] = None,
        container_candidate_texts: Optional[List[str]] = None,
        relation_statement_candidate_texts: Optional[List[str]] = None,
    ):
        self.rng = rng or random.Random(42)

        self.node_candidate_texts = unique_preserve_order((node_candidate_texts or absence_candidate_texts) or [])
        self.absence_candidate_texts = unique_preserve_order(absence_candidate_texts or self.node_candidate_texts)
        self.edge_label_candidate_texts = unique_preserve_order(edge_label_candidate_texts or [])
        self.container_candidate_texts = unique_preserve_order(container_candidate_texts or [])
        self.relation_statement_candidate_texts = unique_preserve_order(relation_statement_candidate_texts or [])

        self.hidden_abstain_unanswerable_ratio = max(0.0, min(1.0, hidden_abstain_unanswerable_ratio))
        self._display_map_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._question_validator = QuestionSafetyValidator()

    # =========================================================
    # Text utilities
    # =========================================================

    def _clean_text(self, text: Optional[str]) -> str:
        return normalize_ws(text or "")

    def _text_token_count(self, text: str) -> int:
        return len([t for t in re.split(r"[\s\-/_,:;()]+", text.lower()) if t])

    def _text_char_length(self, text: str) -> int:
        return len(self._clean_text(text))

    def _has_hyphen(self, text: str) -> bool:
        return "-" in (text or "")

    def _suffix_type(self, text: str) -> str:
        text = self._clean_text(text)
        if not text:
            return "other"
        last = text.split()[-1].lower()
        suffix_map = {
            "block": "block", "layer": "layer", "module": "module",
            "network": "network", "encoding": "encoding", "embedding": "embedding",
            "attention": "attention", "head": "head", "decoder": "decoder",
            "encoder": "encoder", "projection": "projection", "router": "router",
            "routing": "routing", "mask": "mask", "score": "score", "scores": "score",
            "state": "state", "states": "state", "token": "token", "tokens": "token",
            "length": "length",
        }
        return suffix_map.get(last, "other")

    def _infer_domain_hint_from_texts(self, texts: List[str]) -> Set[str]:
        bag = set()
        for txt in texts:
            for t in re.split(r"[\s\-/_,:;()]+", (txt or "").lower()):
                if t and len(t) >= 3:
                    bag.add(t)
        stop = {"the", "and", "for", "with", "from", "into", "over",
                "this", "that", "these", "those", "are", "is", "be"}
        bag -= stop
        architecture_kw = {
            "encoder", "decoder", "attention", "embedding", "token",
            "transformer", "layer", "projection", "head", "block",
            "router", "routing", "expert", "ffn", "rope",
        }
        evaluation_kw = {
            "evaluation", "metric", "metrics", "benchmark", "judge",
            "scenario", "response", "score", "scores", "criteria",
        }
        system_kw = {
            "retrieval", "memory", "cache", "database", "query",
            "pipeline", "system", "storage", "index",
        }
        out = set()
        if bag & architecture_kw:
            out.add("architecture")
        if bag & evaluation_kw:
            out.add("evaluation")
        if bag & system_kw:
            out.add("system")
        return out or {"unknown"}

    def _extract_keywords(self, text: str) -> Set[str]:
        toks = {
            t for t in re.split(r"[\s\-/_,:;()]+", text.lower())
            if t and len(t) >= 3
        }
        stop = {"the", "and", "for", "with", "from", "into", "over",
                "this", "that", "these", "those", "are", "is", "be"}
        return toks - stop

    def _text_style_signature(self, text: str) -> Dict[str, Any]:
        text = self._clean_text(text)
        return {
            "token_count": self._text_token_count(text),
            "char_length": self._text_char_length(text),
            "has_hyphen": self._has_hyphen(text),
            "suffix_type": self._suffix_type(text),
            "keywords": self._extract_keywords(text),
        }

    # =========================================================
    # Basic graph adapters
    # =========================================================

    def _all_entities(self, graph: DiagramGraph) -> List[dict]:
        return graph.all_entities()

    def _all_relations(self, graph: DiagramGraph) -> List[dict]:
        return graph.all_relations()

    def _entity_by_id(self, graph: DiagramGraph, entity_id: str) -> Optional[dict]:
        return graph.get_entity(entity_id)

    def _get_outgoing(self, graph: DiagramGraph, entity_id: str,
                      predicates: Optional[Set[str]] = None,
                      include_conflict: bool = False) -> List[dict]:
        rels = graph.get_outgoing(entity_id, predicate=None, include_conflict=include_conflict)
        if predicates is not None:
            rels = [r for r in rels if r.get("predicate") in predicates]
        return rels

    def _get_incoming(self, graph: DiagramGraph, entity_id: str,
                      predicates: Optional[Set[str]] = None,
                      include_conflict: bool = False) -> List[dict]:
        rels = graph.get_incoming(entity_id, predicate=None, include_conflict=include_conflict)
        if predicates is not None:
            rels = [r for r in rels if r.get("predicate") in predicates]
        return rels

    def _get_containers_of(self, graph: DiagramGraph, entity_id: str) -> List[str]:
        return [str(x) for x in graph.get_parent_containers(entity_id, include_conflict=False)]

    def _has_parent_container(self, graph: DiagramGraph, entity_id: str) -> bool:
        return len(self._get_incoming(graph, entity_id, predicates={"contains"}, include_conflict=False)) > 0

    def _is_top_level_entity(self, graph: DiagramGraph, entity_id: str) -> bool:
        return not self._has_parent_container(graph, entity_id)

    # =========================================================
    # Text safety / style / ranking
    # =========================================================

    def _text_is_safe_for_question(self, text: str) -> bool:
        text = normalize_ws(text)
        if not text:
            return False
        if len(text) > 90:
            return False
        if text.count(" inside ") > 1:
            return False
        if text.count(" immediately after ") > 1:
            return False
        if text.count(" immediately before ") > 1:
            return False
        if text.count(" between ") > 1:
            return False
        return True

    def _token_signature(self, text: str) -> Dict[str, Any]:
        toks = tokenize(text)
        return {
            "num_tokens": len(toks),
            "num_chars": len(text),
            "has_digit": any(ch.isdigit() for ch in text),
            "has_amp": "&" in text,
            "has_hyphen": "-" in text,
            "has_paren": "(" in text or ")" in text,
            "is_title_like": sum(1 for t in text.split() if t[:1].isupper()) >= max(1, len(text.split()) // 2),
            "suffix": toks[-1] if toks else "",
            "prefix": toks[0] if toks else "",
        }

    def _extract_graph_keywords(self, graph: DiagramGraph, topk: int = 12) -> List[str]:
        bag = []
        summary_text = normalize_ws(getattr(graph, "summary_text", "") or "")
        bag.extend([t for t in tokenize(summary_text) if t not in STOPWORDS and len(t) >= 3])
        for ent in self._all_entities(graph):
            txt = normalize_ws(ent.get("text", ""))
            bag.extend([t for t in tokenize(txt) if t not in STOPWORDS and len(t) >= 3])
        if not bag:
            return []
        cnt = Counter(bag)
        return [w for w, _ in cnt.most_common(topk)]

    def _validate_or_drop(self, item, graph):
        if item is None:
            return None
        ok, reason = self._question_validator.validate(item, graph, pg=self)
        if not ok:
            print(f"[DEBUG validator] drop {item.get('question_type')} because: {reason}")
            return None
        return item

    def _graph_domain_hint(self, graph: DiagramGraph) -> Set[str]:
        kws = set(self._extract_graph_keywords(graph, topk=12))
        if not kws:
            return {"unknown"}
        architecture_kw = {
            "encoder", "decoder", "attention", "embedding", "token",
            "transformer", "layer", "projection", "head", "block",
            "router", "routing", "expert", "ffn", "rope",
        }
        evaluation_kw = {
            "evaluation", "metric", "metrics", "benchmark", "judge",
            "scenario", "response", "score", "scores", "criteria",
        }
        system_kw = {
            "retrieval", "memory", "cache", "database", "query",
            "pipeline", "system", "storage", "index",
        }
        out = set()
        if kws & architecture_kw:
            out.add("architecture")
        if kws & evaluation_kw:
            out.add("evaluation")
        if kws & system_kw:
            out.add("system")
        return out or {"unknown"}

    def _graph_node_text_exclude_set(self, graph: DiagramGraph) -> Set[str]:
        texts = unique_preserve_order(
            self._node_display_pool_same_graph(graph) +
            self._present_raw_texts_same_graph(graph)
        )
        return {norm_key(x) for x in texts if self._clean_text(x)}

    def _graph_edge_text_exclude_set(self, graph: DiagramGraph) -> Set[str]:
        texts = []
        for e in getattr(graph, "edges", []) or []:
            txt = self._clean_text(getattr(e, "text", None) or getattr(e, "label", None))
            if txt:
                texts.append(txt)
        return {norm_key(x) for x in texts if x}

    def _candidate_score(self, target_text: str, candidate_text: str,
                         domain_hint: Set[str],
                         candidate_domain_hint: Optional[Set[str]] = None) -> float:
        t = self._text_style_signature(target_text)
        c = self._text_style_signature(candidate_text)
        score = 0.0
        score -= abs(t["token_count"] - c["token_count"]) * 1.2
        score -= abs(t["char_length"] - c["char_length"]) * 0.08
        if t["has_hyphen"] == c["has_hyphen"]:
            score += 0.8
        if t["suffix_type"] == c["suffix_type"] and t["suffix_type"] != "other":
            score += 2.5
        elif t["suffix_type"] != "other" and c["suffix_type"] != "other":
            score += 0.5
        overlap = len(t["keywords"] & c["keywords"])
        score += overlap * 0.8
        if candidate_domain_hint and (domain_hint & candidate_domain_hint):
            score += 1.5
        if c["token_count"] <= 6:
            score += 0.5
        else:
            score -= 0.5
        return score

    def _rank_candidates(self, target_text: str, candidates: List[str],
                         domain_hint: Set[str],
                         exclude_texts: Optional[List[str]] = None) -> List[str]:
        exclude = {norm_key(x) for x in (exclude_texts or []) if normalize_ws(x)}
        cleaned = []
        for c in candidates:
            c = self._clean_text(c)
            if not c or norm_key(c) in exclude:
                continue
            if not self._text_is_safe_for_question(c):
                continue
            cleaned.append(c)
        scored = []
        for c in cleaned:
            c_domain = self._infer_domain_hint_from_texts([c])
            s = self._candidate_score(target_text=target_text, candidate_text=c,
                                      domain_hint=domain_hint, candidate_domain_hint=c_domain)
            scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return unique_preserve_order([c for _, c in scored])

    def _pick_top_k_diverse(self, ranked_candidates: List[str], k: int,
                            target_text: Optional[str] = None) -> List[str]:
        out = []
        used_suffix = set()
        used_prefix = set()
        target_sig = self._text_style_signature(target_text) if target_text else None
        for c in ranked_candidates:
            sig = self._token_signature(c)
            suffix = sig["suffix"]
            prefix = sig["prefix"]
            if target_sig is not None:
                c_sig = self._text_style_signature(c)
                if abs(c_sig["token_count"] - target_sig["token_count"]) > 3:
                    continue
                if abs(c_sig["char_length"] - target_sig["char_length"]) > max(12, target_sig["char_length"]):
                    continue
            if len(out) < max(1, k // 2):
                out.append(c)
                used_suffix.add(suffix)
                used_prefix.add(prefix)
            else:
                if suffix in used_suffix and prefix in used_prefix:
                    continue
                out.append(c)
                used_suffix.add(suffix)
                used_prefix.add(prefix)
            if len(out) >= k:
                break
        return out

    # =========================================================
    # Strict disambiguation
    # =========================================================

    def _raw_entity_text(self, graph: DiagramGraph, entity_id: str) -> str:
        ent = self._entity_by_id(graph, entity_id)
        if not ent:
            return ""
        return normalize_ws(ent.get("text", ""))

    def _raw_text_count(self, graph: DiagramGraph, text: str) -> int:
        if hasattr(graph, "get_text_count") and callable(graph.get_text_count):
            try:
                return int(graph.get_text_count(text))
            except Exception:
                pass
        key = norm_key(text)
        cnt = 0
        for ent in self._all_entities(graph):
            if norm_key(normalize_ws(ent.get("text", ""))) == key:
                cnt += 1
        return cnt

    def _is_raw_text_unique_in_graph(self, graph: DiagramGraph, text: str) -> bool:
        text = normalize_ws(text)
        if not text:
            return False
        return self._raw_text_count(graph, text) == 1

    def _strict_reference_text(self, graph: DiagramGraph, ref_entity_id: str,
                                forbidden_group_key: str) -> Optional[str]:
        txt = self._raw_entity_text(graph, ref_entity_id)
        if not txt:
            return None
        if norm_key(txt) == forbidden_group_key:
            return None
        if not self._is_raw_text_unique_in_graph(graph, txt):
            return None
        if not self._text_is_safe_for_question(txt):
            return None
        return txt

    def _candidate_disambiguation_texts_strict(self, graph: DiagramGraph, entity_id: str) -> List[str]:
        base = self._raw_entity_text(graph, entity_id)
        if not base:
            return []
        group_key = norm_key(base)
        incoming_flow = self._get_incoming(graph, entity_id, predicates={"flows_to"}, include_conflict=False)
        outgoing_flow = self._get_outgoing(graph, entity_id, predicates={"flows_to"}, include_conflict=False)
        container_ids = self._get_containers_of(graph, entity_id)
        pred_texts, succ_texts, container_texts = [], [], []
        for r in incoming_flow:
            txt = self._strict_reference_text(graph, str(r["subject_id"]), group_key)
            if txt:
                pred_texts.append(txt)
        for r in outgoing_flow:
            txt = self._strict_reference_text(graph, str(r["object_id"]), group_key)
            if txt:
                succ_texts.append(txt)
        for cid in container_ids:
            txt = self._strict_reference_text(graph, str(cid), group_key)
            if txt:
                container_texts.append(txt)
        pred_texts = unique_preserve_order(pred_texts)
        succ_texts = unique_preserve_order(succ_texts)
        container_texts = unique_preserve_order(container_texts)
        candidates = []
        for c in container_texts:
            candidates.append(f"{base} inside {c}")
        for p in pred_texts:
            candidates.append(f"{base} immediately after {p}")
        for s in succ_texts:
            candidates.append(f"{base} immediately before {s}")
        for p in pred_texts:
            for s in succ_texts:
                if norm_key(p) != norm_key(s):
                    candidates.append(f"{base} between {p} and {s}")
        return [x for x in unique_preserve_order(candidates) if self._text_is_safe_for_question(x)]

    def _candidate_disambiguation_with_refs(self, graph, entity_id):
        base = self._raw_entity_text(graph, entity_id)
        if not base:
            return []
        group_key = norm_key(base)
        incoming_flow = self._get_incoming(graph, entity_id, predicates={"flows_to"}, include_conflict=False)
        outgoing_flow = self._get_outgoing(graph, entity_id, predicates={"flows_to"}, include_conflict=False)
        container_ids = self._get_containers_of(graph, entity_id)
        pred_refs = []
        for r in incoming_flow:
            pid = str(r["subject_id"])
            txt = self._strict_reference_text(graph, pid, group_key)
            if txt:
                pred_refs.append((txt, pid))
        succ_refs = []
        for r in outgoing_flow:
            oid = str(r["object_id"])
            txt = self._strict_reference_text(graph, oid, group_key)
            if txt:
                succ_refs.append((txt, oid))
        container_refs = []
        for cid in container_ids:
            cid_s = str(cid)
            txt = self._strict_reference_text(graph, cid_s, group_key)
            if txt:
                container_refs.append((txt, cid_s))
        seen_text_keys = set()
        results = []

        def add(phrase, ref_ids):
            k = norm_key(phrase)
            if not k or k in seen_text_keys:
                return
            if not self._text_is_safe_for_question(phrase):
                return
            seen_text_keys.add(k)
            results.append((phrase, set(ref_ids)))

        for c_txt, c_id in container_refs:
            add(f"{base} inside {c_txt}", [c_id])
        for p_txt, p_id in pred_refs:
            add(f"{base} immediately after {p_txt}", [p_id])
        for s_txt, s_id in succ_refs:
            add(f"{base} immediately before {s_txt}", [s_id])
        for p_txt, p_id in pred_refs:
            for s_txt, s_id in succ_refs:
                if norm_key(p_txt) != norm_key(s_txt):
                    add(f"{base} between {p_txt} and {s_txt}", [p_id, s_id])
        return results

    def _build_graph_display_map(self, graph):
        cache_key = id(graph)
        if cache_key in self._display_map_cache:
            return self._display_map_cache[cache_key]

        entities = self._all_entities(graph)
        group2ids = defaultdict(list)
        for ent in entities:
            eid = str(ent.get("id"))
            raw = normalize_ws(ent.get("text", ""))
            group2ids[norm_key(raw)].append(eid)

        result = {}
        for ent in entities:
            eid = str(ent.get("id"))
            raw = normalize_ws(ent.get("text", ""))
            result[eid] = {
                "raw_text": raw,
                "display_text": None,
                "usable": False,
                "used_disambiguation": False,
                "duplicate_group_key": norm_key(raw),
                "referenced_ids": set(),
            }

        for gk, ids in group2ids.items():
            if not gk:
                continue
            if len(ids) == 1:
                eid = ids[0]
                raw = result[eid]["raw_text"]
                if raw and self._text_is_safe_for_question(raw):
                    result[eid].update({
                        "display_text": raw,
                        "usable": True,
                        "used_disambiguation": False,
                        "referenced_ids": set(),
                    })

        for gk, ids in group2ids.items():
            if not gk or len(ids) <= 1:
                continue
            eid2cands = {}
            cand2owners = defaultdict(list)
            for eid in ids:
                cands_with_refs = self._candidate_disambiguation_with_refs(graph, eid)
                cands_with_refs = [
                    (c, r) for (c, r) in cands_with_refs if norm_key(c) != gk
                ]
                eid2cands[eid] = cands_with_refs
                for c, _refs in cands_with_refs:
                    cand2owners[norm_key(c)].append(eid)

            if any(len(eid2cands[eid]) == 0 for eid in ids):
                for eid in ids:
                    result[eid].update({
                        "display_text": None, "usable": False,
                        "used_disambiguation": False, "referenced_ids": set(),
                    })
                continue

            chosen = {}
            chosen_refs = {}
            used_ck = set()
            ok = True
            for eid in sorted(ids, key=lambda x: len(eid2cands[x])):
                picked = None
                picked_refs = set()
                for cand, refs in eid2cands[eid]:
                    ck = norm_key(cand)
                    if cand2owners[ck] == [eid] and ck not in used_ck:
                        picked = cand
                        picked_refs = refs
                        break
                if picked is None:
                    ok = False
                    break
                chosen[eid] = picked
                chosen_refs[eid] = picked_refs
                used_ck.add(norm_key(picked))

            if not ok or len(chosen) != len(ids):
                for eid in ids:
                    result[eid].update({
                        "display_text": None, "usable": False,
                        "used_disambiguation": False, "referenced_ids": set(),
                    })
                continue

            for eid in ids:
                result[eid].update({
                    "display_text": chosen[eid],
                    "usable": True,
                    "used_disambiguation": True,
                    "referenced_ids": chosen_refs[eid],
                })

        self._display_map_cache[cache_key] = result
        return result

    def _display_text_or_none(self, graph: DiagramGraph, entity_id: str) -> Optional[str]:
        info = self._build_graph_display_map(graph).get(str(entity_id))
        if not info or not info["usable"]:
            return None
        return info["display_text"]

    def _entity_display_text_and_flag(self, graph: DiagramGraph, entity_id: str) -> Tuple[Optional[str], bool]:
        info = self._build_graph_display_map(graph).get(str(entity_id))
        if not info or not info["usable"]:
            return None, False
        return info["display_text"], bool(info["used_disambiguation"])

    def _usable_entity_ids(self, graph: DiagramGraph) -> List[str]:
        return [eid for eid, info in self._build_graph_display_map(graph).items()
                if info["usable"] and info["display_text"]]

    def _top_level_usable_entity_ids(self, graph: DiagramGraph) -> List[str]:
        return [eid for eid in self._usable_entity_ids(graph) if self._is_top_level_entity(graph, eid)]

    def _node_display_pool_same_graph(self, graph: DiagramGraph) -> List[str]:
        out = []
        for eid in self._usable_entity_ids(graph):
            txt = self._display_text_or_none(graph, eid)
            if txt:
                out.append(txt)
        return unique_preserve_order(out)

    def _present_raw_texts_same_graph(self, graph: DiagramGraph) -> List[str]:
        out = []
        for ent in self._all_entities(graph):
            txt = normalize_ws(ent.get("text", ""))
            if txt:
                out.append(txt)
        return unique_preserve_order(out)

    def _present_raw_keys_same_graph(self, graph: DiagramGraph) -> Set[str]:
        return {norm_key(t) for t in self._present_raw_texts_same_graph(graph)}

    # =========================================================
    # Option helpers
    # =========================================================

    def _shuffle_options(self, option_texts: List[str],
                         correct_texts: List[str]) -> Tuple[List[Dict[str, str]], List[int], List[str]]:
        option_texts = unique_preserve_order(option_texts)
        correct_set = {norm_key(x) for x in correct_texts}
        self.rng.shuffle(option_texts)
        options = []
        correct_indices = []
        for i, txt in enumerate(option_texts):
            label = OPTION_LABELS[i]
            options.append({"label": label, "text": txt})
            if norm_key(txt) in correct_set:
                correct_indices.append(i)
        answer_labels = [options[i]["label"] for i in correct_indices]
        return options, correct_indices, answer_labels

    def _validate_single_choice_options(self, options: List[Dict[str, str]], correct_indices: List[int]) -> bool:
        if len(options) != 4:
            return False
        if len(correct_indices) != 1:
            return False
        texts = [norm_key(x["text"]) for x in options]
        return len(set(texts)) == 4

    def _validate_multi_select_options(self, options: List[Dict[str, str]], correct_indices: List[int]) -> bool:
        if len(options) < 4:
            return False
        if len(correct_indices) < 2:
            return False
        texts = [norm_key(x["text"]) for x in options]
        return len(set(texts)) == len(texts)

    def _build_sample(self, graph: DiagramGraph, level: str, qtype: str,
                      question: str, answer_type: str, answer: Any, answer_text: Any,
                      options: Optional[List[Dict[str, str]]] = None,
                      correct_option_indices: Optional[List[int]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sample = {
            "image_id": getattr(graph, "image_id", ""),
            "paper_id": getattr(graph, "paper_id", ""),
            "level": level,
            "question_type": qtype,
            "question": question,
            "answer_type": answer_type,
            "answer": answer,
            "options": options if options is not None else [],
            "answer_text": answer_text,
        }
        if correct_option_indices is not None:
            sample["correct_option_indices"] = correct_option_indices
        if metadata is not None:
            sample["metadata"] = metadata
        return sample

    # =========================================================
    # Distractor sampling
    # =========================================================

    def _sample_presence_absent_distractors(self, graph: DiagramGraph,
                                            target_text: str, k: int) -> List[str]:
        present_raw_keys = self._present_raw_keys_same_graph(graph)
        ranked = self._rank_candidates(
            target_text=target_text,
            candidates=(self.node_candidate_texts or self.absence_candidate_texts),
            domain_hint=self._graph_domain_hint(graph),
            exclude_texts=self._present_raw_texts_same_graph(graph) + [target_text],
        )
        filtered = [x for x in ranked if norm_key(x) not in present_raw_keys]
        return self._pick_top_k_diverse(filtered, k, target_text=target_text)

    def _sample_answerable_node_distractors(self, graph: DiagramGraph, target_text: str,
                                            exclude_texts: Optional[List[str]] = None,
                                            k: int = 3) -> List[str]:
        ranked = self._rank_candidates(
            target_text=target_text,
            candidates=self._node_display_pool_same_graph(graph),
            domain_hint=self._graph_domain_hint(graph),
            exclude_texts=exclude_texts or [target_text],
        )
        return self._pick_top_k_diverse(ranked, k)

    def _sample_hidden_abstain_node_options(self, graph: DiagramGraph, source_text: str,
                                            forbidden_texts: List[str], k: int = 4) -> List[str]:
        ranked = self._rank_candidates(
            target_text=source_text,
            candidates=self._node_display_pool_same_graph(graph),
            domain_hint=self._graph_domain_hint(graph),
            exclude_texts=forbidden_texts,
        )
        return self._pick_top_k_diverse(ranked, k)[:k]

    def _sample_edge_label_distractors(self, graph: DiagramGraph, target_text: str,
                                       exclude_texts: Optional[List[str]] = None,
                                       k: int = 3) -> List[str]:
        ranked = self._rank_candidates(
            target_text=target_text,
            candidates=self.edge_label_candidate_texts,
            domain_hint=self._graph_domain_hint(graph),
            exclude_texts=exclude_texts or [target_text],
        )
        return self._pick_top_k_diverse(ranked, k)

    # =========================================================
    # Multihop helpers
    # =========================================================

    def _entity_container_set(self, graph: DiagramGraph, entity_id: str) -> Set[str]:
        return set(self._get_containers_of(graph, entity_id))

    def _shared_container_for_path(self, graph: DiagramGraph, path: List[str]) -> Optional[str]:
        if not path:
            return None
        container_sets = []
        for eid in path:
            cset = self._entity_container_set(graph, eid)
            if not cset:
                return None
            container_sets.append(cset)
        shared = set.intersection(*container_sets) if container_sets else set()
        return sorted(shared)[0] if shared else None

    def _usable_entities_in_container(self, graph: DiagramGraph, container_id: str) -> List[str]:
        usable = set(self._usable_entity_ids(graph))
        return [str(cid) for cid in graph.get_contained_children(container_id, include_conflict=False)
                if str(cid) in usable]

    def _build_scope_adjacency(self, graph: DiagramGraph,
                               allowed_node_ids: Set[str]) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = defaultdict(list)
        for eid in allowed_node_ids:
            for r in self._get_outgoing(graph, eid, predicates={"flows_to"}, include_conflict=False):
                obj = str(r["object_id"])
                if obj in allowed_node_ids:
                    adj[str(eid)].append(obj)
        return adj

    def _enumerate_simple_paths_limited(self, adj: Dict[str, List[str]], src: str, tgt: str,
                                        max_hops: int, max_paths: int = 2) -> List[List[str]]:
        results: List[List[str]] = []

        def dfs(cur: str, path: List[str], visited: Set[str]):
            if len(results) >= max_paths or len(path) - 1 > max_hops:
                return
            if cur == tgt:
                results.append(path[:])
                return
            for nxt in adj.get(cur, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, path, visited)
                path.pop()
                visited.remove(nxt)

        dfs(src, [src], {src})
        return results

    def _all_unique_multihop_pairs_in_scope(self, graph: DiagramGraph,
                                            allowed_node_ids: Set[str],
                                            min_hops: int = 2,
                                            max_hops: int = 4) -> List[Tuple[str, str, List[str]]]:
        candidates = list(allowed_node_ids)
        if len(candidates) < 3:
            return []
        adj = self._build_scope_adjacency(graph, allowed_node_ids)
        results = []
        for src in candidates:
            for tgt in candidates:
                if src == tgt:
                    continue
                paths = self._enumerate_simple_paths_limited(adj=adj, src=src, tgt=tgt,
                                                             max_hops=max_hops, max_paths=2)
                if len(paths) != 1:
                    continue
                path = paths[0]
                hops = len(path) - 1
                if min_hops <= hops <= max_hops:
                    results.append((src, tgt, path))
        return results

    def _build_multihop_distractors(self, graph: DiagramGraph,
                                    target_intermediate_displays: List[str],
                                    path: List[str],
                                    primary_scope_ids: Set[str],
                                    fallback_to_same_graph: bool = True,
                                    min_wrong: int = 2) -> List[str]:
        if not target_intermediate_displays:
            return []
        path_set = set(path)
        picked: List[str] = []
        hard_exclude_keys: Set[str] = {
            norm_key(x) for x in target_intermediate_displays if self._clean_text(x)
        }
        for eid in path:
            txt = self._display_text_or_none(graph, eid)
            if self._clean_text(txt):
                hard_exclude_keys.add(norm_key(txt))
        style_target = target_intermediate_displays[0]

        def try_add(text: Optional[str]) -> None:
            text = self._clean_text(text)
            if not text or norm_key(text) in hard_exclude_keys:
                return
            if any(norm_key(x) == norm_key(text) for x in picked):
                return
            picked.append(text)

        scope_candidates = [
            self._clean_text(self._display_text_or_none(graph, eid))
            for eid in primary_scope_ids if eid not in path_set
        ]
        scope_candidates = [x for x in scope_candidates if x and norm_key(x) not in hard_exclude_keys]
        for x in self._pick_top_k_diverse(
            self._rank_candidates(style_target, scope_candidates,
                                  self._graph_domain_hint(graph), list(hard_exclude_keys)),
            min_wrong, target_text=style_target
        ):
            try_add(x)
        if len(picked) >= min_wrong:
            return picked[:min_wrong]

        if fallback_to_same_graph:
            graph_candidates = [
                self._clean_text(self._display_text_or_none(graph, eid))
                for eid in self._usable_entity_ids(graph) if eid not in path_set
            ]
            graph_candidates = [x for x in graph_candidates if x and norm_key(x) not in hard_exclude_keys]
            for x in self._pick_top_k_diverse(
                self._rank_candidates(style_target, graph_candidates,
                                      self._graph_domain_hint(graph),
                                      list(hard_exclude_keys) + picked),
                min_wrong - len(picked), target_text=style_target
            ):
                try_add(x)
            if len(picked) >= min_wrong:
                return picked[:min_wrong]

        present_raw_keys = self._present_raw_keys_same_graph(graph)
        for x in self._pick_top_k_diverse(
            self._rank_candidates(
                style_target,
                (self.node_candidate_texts or self.absence_candidate_texts),
                self._graph_domain_hint(graph),
                list(hard_exclude_keys | present_raw_keys) + picked,
            ),
            min_wrong - len(picked), target_text=style_target
        ):
            try_add(x)

        return picked[:min_wrong]

    def _can_build_multihop_options_for_scope(self, graph: DiagramGraph,
                                              path: List[str], scope_ids: Set[str]) -> bool:
        if len(path) < 3:
            return False
        intermediate_ids = path[1:-1]
        if not intermediate_ids:
            return False
        intermediate_displays = []
        for eid in intermediate_ids:
            disp = self._display_text_or_none(graph, eid)
            if disp is None:
                return False
            intermediate_displays.append(disp)
        required_wrong = max(1, 4 - len(intermediate_displays))
        distractors = self._build_multihop_distractors(
            graph=graph, target_intermediate_displays=intermediate_displays,
            path=path, primary_scope_ids=scope_ids,
            fallback_to_same_graph=True, min_wrong=required_wrong,
        )
        return len(unique_preserve_order(intermediate_displays + distractors)) >= 4

    def _all_edge_texts_same_graph(self, graph: DiagramGraph) -> List[str]:
        texts = []
        for e in getattr(graph, "edges", []) or []:
            txt = self._clean_text(getattr(e, "text", None) or getattr(e, "label", None))
            if txt and self._text_is_safe_for_question(txt):
                texts.append(txt)
        return unique_preserve_order(texts)

    # =========================================================
    # Multihop: shared emission helper
    # =========================================================

    def _build_and_emit_multihop_sample(self, graph, src_id, tgt_id, path,
                                        scope_ids, path_scope,
                                        container_display=None, container_id=None):
        src_display, src_disamb = self._entity_display_text_and_flag(graph, src_id)
        tgt_display, tgt_disamb = self._entity_display_text_and_flag(graph, tgt_id)
        if src_display is None or tgt_display is None:
            return None

        intermediate_ids = path[1:-1]
        if not intermediate_ids:
            return None

        intermediate_displays = []
        used_disamb = bool(src_disamb or tgt_disamb)
        for eid in intermediate_ids:
            disp, disamb = self._entity_display_text_and_flag(graph, eid)
            if disp is None:
                return None
            intermediate_displays.append(disp)
            used_disamb = bool(used_disamb or disamb)

        # GUARD A
        display_map = self._build_graph_display_map(graph)
        src_refs = display_map.get(str(src_id), {}).get("referenced_ids", set())
        tgt_refs = display_map.get(str(tgt_id), {}).get("referenced_ids", set())
        intermediate_id_set = {str(x) for x in intermediate_ids}
        if (src_refs & intermediate_id_set) or (tgt_refs & intermediate_id_set):
            return None

        # GUARD B
        src_disp_key = norm_key(src_display)
        tgt_disp_key = norm_key(tgt_display)
        for intm_txt in intermediate_displays:
            intm_key = norm_key(intm_txt)
            if not intm_key:
                continue
            if intm_key in src_disp_key or intm_key in tgt_disp_key:
                return None

        is_single = (len(intermediate_ids) == 1)
        n_correct = len(intermediate_displays)
        min_wrong = max(1, 4 - n_correct)

        distractors = self._build_multihop_distractors(
            graph=graph, target_intermediate_displays=intermediate_displays,
            path=path, primary_scope_ids=scope_ids,
            fallback_to_same_graph=True, min_wrong=min_wrong,
        )

        options_texts = unique_preserve_order(intermediate_displays + distractors)
        bad_keys = {norm_key(src_display), norm_key(tgt_display)}
        options_texts = [x for x in options_texts if norm_key(x) not in bad_keys]

        if len(options_texts) < 4:
            return None

        max_options = min(6, max(4, n_correct + 2))
        if len(options_texts) > max_options:
            correct_key_set = {norm_key(y) for y in intermediate_displays}
            wrongs = [x for x in options_texts if norm_key(x) not in correct_key_set]
            self.rng.shuffle(wrongs)
            options_texts = intermediate_displays + wrongs[:max_options - n_correct]
            options_texts = unique_preserve_order(options_texts)

        options_texts = [x for x in options_texts if norm_key(x) not in bad_keys]
        if len(options_texts) < 4:
            return None

        options, correct_indices, answer_labels = self._shuffle_options(
            options_texts, intermediate_displays)

        if is_single:
            if not self._validate_single_choice_options(options, correct_indices):
                return None
            answer_type = "single_choice"
            answer_value = answer_labels[0]
            answer_text_value = intermediate_displays[0]
        else:
            if not self._validate_multi_select_options(options, correct_indices):
                return None
            answer_type = "multi_select"
            answer_value = answer_labels
            answer_text_value = intermediate_displays

        if path_scope == "top_level_only":
            question = self.rng.choice([
                'Considering only the top-level pipeline stages, select all intermediate stages on the unique path from "{src}" to "{tgt}".',
                'Select all intermediate top-level stages on the unique flow path from "{src}" to "{tgt}".',
                'On the unique path from "{src}" to "{tgt}", which top-level pipeline stages appear in between? Select all that apply.',
            ]).format(src=src_display, tgt=tgt_display)
        else:
            question = self.rng.choice([
                'Within "{container}", select all intermediate stages on the unique flow path from "{src}" to "{tgt}".',
                'Inside "{container}", which main blocks appear as intermediate stages on the unique path from "{src}" to "{tgt}"? Select all that apply.',
            ]).format(container=container_display, src=src_display, tgt=tgt_display)

        metadata = {
            "used_disambiguation": used_disamb,
            "path_scope": path_scope,
            "path_unique": True,
            "src_id": src_id, "tgt_id": tgt_id,
            "src_display": src_display, "tgt_display": tgt_display,
            "path_node_ids": list(path),
            "intermediate_node_ids": list(intermediate_ids),
            "intermediate_displays": list(intermediate_displays),
        }
        if path_scope == "local_container":
            metadata["scope_container"] = container_display
            metadata["scope_container_id"] = container_id

        item = self._build_sample(
            graph=graph, level="L2", qtype="relation_multihop_multiselect",
            question=question, options=options, answer=answer_value,
            answer_text=answer_text_value, correct_option_indices=correct_indices,
            answer_type=answer_type, metadata=metadata,
        )
        return self._validate_or_drop(item, graph)

    # =========================================================
    # Question generators
    # =========================================================

    def generate_element_presence_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        display_map = self._build_graph_display_map(graph)
        usable_entities = [eid for eid, info in display_map.items()
                           if info["usable"] and info["display_text"]]
        if not usable_entities:
            return None
        target_id = self.rng.choice(usable_entities)
        correct = display_map[target_id]["display_text"]
        used_disamb = display_map[target_id]["used_disambiguation"]
        distractors = self._sample_presence_absent_distractors(graph=graph, target_text=correct, k=3)
        if len(distractors) < 3:
            return None
        options, correct_indices, answer_labels = self._shuffle_options(
            [correct] + distractors[:3], [correct])
        if not self._validate_single_choice_options(options, correct_indices):
            return None
        question = self.rng.choice([
            "Which of the following labels appears in the diagram?",
            "Which label can be found in this figure?",
            "Select the label that is present in the diagram.",
        ])
        item = self._build_sample(
            graph=graph, level="L1", qtype="element_presence_mcq", question=question,
            options=options, answer=answer_labels[0], answer_text=correct,
            correct_option_indices=correct_indices, answer_type="single_choice",
            metadata={"used_disambiguation": used_disamb},
        )
        return self._validate_or_drop(item, graph)

    def generate_element_absence_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        present_display_texts = self._node_display_pool_same_graph(graph)
        present_raw_keys = self._present_raw_keys_same_graph(graph)
        if len(present_display_texts) < 3:
            return None
        seed_target = self.rng.choice(present_display_texts)
        ranked_absent = self._rank_candidates(
            target_text=seed_target,
            candidates=(self.node_candidate_texts or self.absence_candidate_texts),
            domain_hint=self._graph_domain_hint(graph),
            exclude_texts=self._present_raw_texts_same_graph(graph) + [seed_target],
        )
        ranked_absent = [x for x in ranked_absent if norm_key(x) not in present_raw_keys]
        if not ranked_absent:
            return None
        correct = ranked_absent[0]
        distractors = self._pick_top_k_diverse(
            self._rank_candidates(correct, present_display_texts,
                                  self._graph_domain_hint(graph), [correct]),
            3, target_text=correct,
        )
        if len(distractors) < 3:
            return None
        options, correct_indices, answer_labels = self._shuffle_options(
            [correct] + distractors[:3], [correct])
        if not self._validate_single_choice_options(options, correct_indices):
            return None
        question = self.rng.choice([
            "Which of the following labels does NOT appear in the diagram?",
            "Select the label that is absent from this figure.",
            "Which option cannot be found in the diagram?",
        ])
        item = self._build_sample(
            graph=graph, level="L1", qtype="element_absence_mcq", question=question,
            options=options, answer=answer_labels[0], answer_text=correct,
            correct_option_indices=correct_indices, answer_type="single_choice", metadata=None,
        )
        return self._validate_or_drop(item, graph)

    # =========================================================
    # [PLAN C] successor: hidden_abstain (unanswerable = 0 successors only)
    # =========================================================

    def generate_relation_successor_hidden_abstain_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        usable_ids = self._usable_entity_ids(graph)
        if len(usable_ids) < 4:
            return None

        answerable = self.rng.random() > self.hidden_abstain_unanswerable_ratio

        if answerable:
            pairs = [(eid, graph.get_unique_successor(eid, include_conflict=False))
                     for eid in usable_ids]
            pairs = [(s, t) for s, t in pairs if t is not None and t in usable_ids]
            if pairs:
                src_id, tgt_id = self.rng.choice(pairs)
                src_text, src_disamb = self._entity_display_text_and_flag(graph, src_id)
                tgt_text, tgt_disamb = self._entity_display_text_and_flag(graph, tgt_id)
                if src_text is not None and tgt_text is not None:
                    distractors = self._sample_answerable_node_distractors(
                        graph=graph, target_text=tgt_text,
                        exclude_texts=[src_text, tgt_text], k=3)
                    if len(distractors) >= 3:
                        options, correct_indices, answer_labels = self._shuffle_options(
                            [tgt_text] + distractors[:3], [tgt_text])
                        if self._validate_single_choice_options(options, correct_indices):
                            question = self.rng.choice([
                                'According to the arrows in the diagram, which module does "{src}" directly point to?',
                                'Which module is the direct target of the arrow leaving "{src}"?',
                                'Looking at the diagram, which module does the arrow from "{src}" lead to?',
                            ]).format(src=src_text)
                            return self._build_sample(
                                graph=graph, level="L2",
                                qtype="relation_successor_hidden_abstain_mcq",
                                question=question, options=options, answer=answer_labels[0],
                                answer_text=tgt_text, correct_option_indices=correct_indices,
                                answer_type="single_choice",
                                metadata={"is_unanswerable": False,
                                          "used_disambiguation": bool(src_disamb or tgt_disamb)},
                            )

        # [PLAN C] Unanswerable: ONLY nodes with 0 successors (terminal nodes)
        # Nodes with >=2 successors are handled by generate_relation_successor_multiselect
        no_successor_sources = []
        for eid in usable_ids:
            succs = [x for x in graph.get_successors(eid, flow_only=True, include_conflict=False)
                     if x in usable_ids]
            if len(succs) == 0:
                no_successor_sources.append(eid)

        if not no_successor_sources:
            return None

        src_id = self.rng.choice(no_successor_sources)
        src_text, src_disamb = self._entity_display_text_and_flag(graph, src_id)
        if src_text is None:
            return None

        forbidden = [src_text]
        options_texts = self._sample_hidden_abstain_node_options(
            graph=graph, source_text=src_text, forbidden_texts=forbidden, k=4)
        if len(options_texts) < 4:
            return None
        options, correct_indices, _ = self._shuffle_options(options_texts[:4], [])
        if correct_indices:
            return None

        question = self.rng.choice([
            'According to the arrows in the diagram, which module does "{src}" directly point to?',
            'Which module is the direct target of the arrow leaving "{src}"?',
            'Looking at the diagram, which module does the arrow from "{src}" lead to?',
        ]).format(src=src_text)

        return self._build_sample(
            graph=graph, level="L2", qtype="relation_successor_hidden_abstain_mcq",
            question=question, options=options, answer="cannot_be_determined",
            answer_text="cannot_be_determined", correct_option_indices=[],
            answer_type="single_choice",
            metadata={"is_unanswerable": True,
                      "unanswerable_reason": "no_usable_successor",
                      "used_disambiguation": bool(src_disamb)},
        )

    # =========================================================
    # [PLAN C] NEW: successor multiselect (>=2 successors)
    # =========================================================

    def generate_relation_successor_multiselect(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        usable_ids = self._usable_entity_ids(graph)
        if len(usable_ids) < 5:
            return None

        candidates = []
        for eid in usable_ids:
            succs = [x for x in graph.get_successors(eid, flow_only=True, include_conflict=False)
                     if x in usable_ids]
            if len(succs) >= 2:
                succ_displays = []
                all_ok = True
                for s in succs:
                    disp = self._display_text_or_none(graph, s)
                    if disp is None:
                        all_ok = False
                        break
                    succ_displays.append((s, disp))
                if all_ok:
                    candidates.append((eid, succ_displays))

        if not candidates:
            return None

        src_id, succ_list = self.rng.choice(candidates)
        src_text, src_disamb = self._entity_display_text_and_flag(graph, src_id)
        if src_text is None:
            return None

        correct_displays = [disp for _, disp in succ_list]
        correct_keys = {norm_key(x) for x in correct_displays}
        used_disamb = bool(src_disamb)

        exclude = [src_text] + correct_displays
        distractors = self._sample_answerable_node_distractors(
            graph=graph, target_text=correct_displays[0],
            exclude_texts=exclude,
            k=max(1, 4 - len(correct_displays)))

        options_texts = unique_preserve_order(correct_displays + distractors)
        options_texts = [x for x in options_texts if norm_key(x) != norm_key(src_text)]

        if len(options_texts) < 4:
            return None

            # 正确答案太多时截断,确保不超过可用标签数
        if len(correct_displays) > len(OPTION_LABELS) - 2:
            correct_displays = correct_displays[:len(OPTION_LABELS) - 2]
            correct_keys = {norm_key(x) for x in correct_displays}

        max_options = min(len(OPTION_LABELS), max(4, len(correct_displays) + 2))
        if len(options_texts) > max_options:
            wrongs = [x for x in options_texts if norm_key(x) not in correct_keys]
            self.rng.shuffle(wrongs)
            options_texts = correct_displays + wrongs[:max_options - len(correct_displays)]
            options_texts = unique_preserve_order(options_texts)

        if len(options_texts) < 4:
            return None

        options, correct_indices, answer_labels = self._shuffle_options(
            options_texts, correct_displays)

        if len(correct_indices) < 2:
            return None
        if not self._validate_multi_select_options(options, correct_indices):
            return None

        question = self.rng.choice([
            'According to the arrows in the diagram, which modules does "{src}" directly point to? Select all that apply.',
            'Which modules are direct targets of arrows leaving "{src}"? Select all that apply.',
            'Looking at the diagram, which modules do the arrows from "{src}" lead to? Select all that apply.',
        ]).format(src=src_text)

        item = self._build_sample(
            graph=graph, level="L2",
            qtype="relation_successor_multiselect",
            question=question, options=options,
            answer=answer_labels, answer_text=correct_displays,
            correct_option_indices=correct_indices,
            answer_type="multi_select",
            metadata={"used_disambiguation": used_disamb,
                      "src_id": src_id,
                      "src_display": src_text,
                      "successor_count": len(correct_displays)},
        )
        return self._validate_or_drop(item, graph)

    # =========================================================
    # [PLAN C] containment: hidden_abstain (unanswerable = 0 children only)
    # =========================================================

    def generate_relation_containment_hidden_abstain_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        usable_ids = self._usable_entity_ids(graph)
        if len(usable_ids) < 4:
            return None

        answerable = self.rng.random() > self.hidden_abstain_unanswerable_ratio

        if answerable:
            pairs = []
            for child_id in usable_ids:
                parent = graph.get_unique_parent_container(child_id, include_conflict=False)
                if parent is not None and parent in usable_ids:
                    pairs.append((parent, child_id))
            if pairs:
                parent_id, child_id = self.rng.choice(pairs)
                parent_text, p_disamb = self._entity_display_text_and_flag(graph, parent_id)
                child_text, c_disamb = self._entity_display_text_and_flag(graph, child_id)
                if parent_text is not None and child_text is not None:
                    distractors = self._sample_answerable_node_distractors(
                        graph=graph, target_text=child_text,
                        exclude_texts=[parent_text, child_text], k=3)
                    if len(distractors) >= 3:
                        options, correct_indices, answer_labels = self._shuffle_options(
                            [child_text] + distractors[:3], [child_text])
                        if self._validate_single_choice_options(options, correct_indices):
                            question = self.rng.choice([
                                'Which module is visually enclosed within the boundary of "{container}" in the diagram?',
                                'Looking at the diagram, which module is drawn inside the box labeled "{container}"?',
                                'Which module appears within the visual boundary of "{container}"?',
                            ]).format(container=parent_text)
                            return self._build_sample(
                                graph=graph, level="L2",
                                qtype="relation_containment_hidden_abstain_mcq",
                                question=question, options=options, answer=answer_labels[0],
                                answer_text=child_text, correct_option_indices=correct_indices,
                                answer_type="single_choice",
                                metadata={"is_unanswerable": False,
                                          "used_disambiguation": bool(p_disamb or c_disamb)},
                            )

        # [PLAN C] Unanswerable: ONLY entities with 0 children
        # Entities with >=2 children are handled by generate_relation_containment_multiselect
        no_child_entities = []
        for eid in usable_ids:
            children = [x for x in graph.get_contained_children(eid, include_conflict=False)
                        if x in usable_ids]
            if len(children) == 0:
                no_child_entities.append(eid)

        if not no_child_entities:
            return None

        container_id = self.rng.choice(no_child_entities)
        container_text, used_disamb = self._entity_display_text_and_flag(graph, container_id)
        if container_text is None:
            return None

        forbidden = [container_text]
        options_texts = self._sample_hidden_abstain_node_options(
            graph=graph, source_text=container_text, forbidden_texts=forbidden, k=4)
        if len(options_texts) < 4:
            return None
        options, correct_indices, _ = self._shuffle_options(options_texts[:4], [])
        if correct_indices:
            return None

        question = self.rng.choice([
            'Which module is visually enclosed within the boundary of "{container}" in the diagram?',
            'Looking at the diagram, which module is drawn inside the box labeled "{container}"?',
            'Which module appears within the visual boundary of "{container}"?',
        ]).format(container=container_text)

        return self._build_sample(
            graph=graph, level="L2", qtype="relation_containment_hidden_abstain_mcq",
            question=question, options=options, answer="cannot_be_determined",
            answer_text="cannot_be_determined", correct_option_indices=[],
            answer_type="single_choice",
            metadata={"is_unanswerable": True,
                      "unanswerable_reason": "no_usable_child",
                      "used_disambiguation": bool(used_disamb)},
        )

    # =========================================================
    # [PLAN C] NEW: containment multiselect (>=2 children)
    # =========================================================

    def generate_relation_containment_multiselect(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        usable_ids = self._usable_entity_ids(graph)
        if len(usable_ids) < 5:
            return None

        candidates = []
        for eid in usable_ids:
            children = [x for x in graph.get_contained_children(eid, include_conflict=False)
                        if x in usable_ids]
            if len(children) >= 2:
                child_displays = []
                all_ok = True
                for c in children:
                    disp = self._display_text_or_none(graph, c)
                    if disp is None:
                        all_ok = False
                        break
                    child_displays.append((c, disp))
                if all_ok:
                    candidates.append((eid, child_displays))

        if not candidates:
            return None

        container_id, child_list = self.rng.choice(candidates)
        container_text, container_disamb = self._entity_display_text_and_flag(graph, container_id)
        if container_text is None:
            return None

        correct_displays = [disp for _, disp in child_list]
        correct_keys = {norm_key(x) for x in correct_displays}
        used_disamb = bool(container_disamb)

        exclude = [container_text] + correct_displays
        distractors = self._sample_answerable_node_distractors(
            graph=graph, target_text=correct_displays[0],
            exclude_texts=exclude,
            k=max(1, 4 - len(correct_displays)))

        options_texts = unique_preserve_order(correct_displays + distractors)
        options_texts = [x for x in options_texts if norm_key(x) != norm_key(container_text)]

        if len(options_texts) < 4:
            return None

        if len(correct_displays) > len(OPTION_LABELS) - 2:
            correct_displays = correct_displays[:len(OPTION_LABELS) - 2]
            correct_keys = {norm_key(x) for x in correct_displays}

        max_options = min(len(OPTION_LABELS), max(4, len(correct_displays) + 2))
        if len(options_texts) > max_options:
            wrongs = [x for x in options_texts if norm_key(x) not in correct_keys]
            self.rng.shuffle(wrongs)
            options_texts = correct_displays + wrongs[:max_options - len(correct_displays)]
            options_texts = unique_preserve_order(options_texts)

        if len(options_texts) < 4:
            return None

        options, correct_indices, answer_labels = self._shuffle_options(
            options_texts, correct_displays)

        if len(correct_indices) < 2:
            return None
        if not self._validate_multi_select_options(options, correct_indices):
            return None

        question = self.rng.choice([
            'Which modules are visually enclosed within the boundary of "{container}" in the diagram? Select all that apply.',
            'Looking at the diagram, which modules are drawn inside the box labeled "{container}"? Select all that apply.',
            'Which modules appear within the visual boundary of "{container}"? Select all that apply.',
        ]).format(container=container_text)

        item = self._build_sample(
            graph=graph, level="L2",
            qtype="relation_containment_multiselect",
            question=question, options=options,
            answer=answer_labels, answer_text=correct_displays,
            correct_option_indices=correct_indices,
            answer_type="multi_select",
            metadata={"used_disambiguation": used_disamb,
                      "container_id": container_id,
                      "container_display": container_text,
                      "child_count": len(correct_displays)},
        )
        return self._validate_or_drop(item, graph)

    # =========================================================
    # Edge label (unchanged)
    # =========================================================

    def generate_relation_edge_label_hidden_abstain_mcq(self, graph):
        if not graph.get_edges_with_text(predicate=None, include_conflict=False):
            return None
        if len(self.edge_label_candidate_texts) < 4:
            return None

        answerable = self.rng.random() > self.hidden_abstain_unanswerable_ratio

        if answerable:
            rels = [
                r for r in graph.get_edges_with_text(predicate=None, include_conflict=False)
                if self._display_text_or_none(graph, str(r["subject_id"])) is not None
                   and self._display_text_or_none(graph, str(r["object_id"])) is not None
            ]
            if rels:
                rel = self.rng.choice(rels)
                correct = normalize_ws(rel.get("edge_text", ""))
                s_text, s_disamb = self._entity_display_text_and_flag(graph, str(rel["subject_id"]))
                o_text, o_disamb = self._entity_display_text_and_flag(graph, str(rel["object_id"]))
                if correct and s_text is not None and o_text is not None:
                    distractors = self._sample_edge_label_distractors(
                        graph=graph, target_text=correct, exclude_texts=[correct], k=3)
                    if len(distractors) >= 3:
                        options, correct_indices, answer_labels = self._shuffle_options(
                            [correct] + distractors[:3], [correct])
                        if self._validate_single_choice_options(options, correct_indices):
                            question = self.rng.choice([
                                'What is the label on the edge from "{src}" to "{tgt}"?',
                                'Which text labels the relation from "{src}" to "{tgt}"?',
                                'Select the edge label between "{src}" and "{tgt}".',
                            ]).format(src=s_text, tgt=o_text)
                            return self._build_sample(
                                graph=graph, level="L2",
                                qtype="relation_edge_label_hidden_abstain_mcq",
                                question=question, options=options, answer=answer_labels[0],
                                answer_text=correct, correct_option_indices=correct_indices,
                                answer_type="single_choice",
                                metadata={"is_unanswerable": False,
                                          "used_disambiguation": bool(s_disamb or o_disamb)},
                            )

        usable_ids = self._usable_entity_ids(graph)
        if len(usable_ids) < 2:
            return None
        candidate_pairs = []
        for _ in range(100):
            a, b = self.rng.sample(usable_ids, 2)
            rels = [
                r for r in self._get_outgoing(graph, a, predicates=None, include_conflict=False)
                if str(r["object_id"]) == b
            ]
            if not rels or all(not normalize_ws(r.get("edge_text", "")) for r in rels):
                candidate_pairs.append((a, b))
        if not candidate_pairs:
            return None
        subj, obj = self.rng.choice(candidate_pairs)
        s_text, s_disamb = self._entity_display_text_and_flag(graph, subj)
        o_text, o_disamb = self._entity_display_text_and_flag(graph, obj)
        if s_text is None or o_text is None:
            return None
        options_texts = self._sample_edge_label_distractors(
            graph=graph,
            target_text=self.rng.choice(self.edge_label_candidate_texts),
            exclude_texts=[], k=4,
        )
        if len(options_texts) < 4:
            return None
        options, correct_indices, _ = self._shuffle_options(options_texts[:4], [])
        if correct_indices:
            return None
        question = self.rng.choice([
            'What is the label on the edge from "{src}" to "{tgt}"?',
            'Which text labels the relation from "{src}" to "{tgt}"?',
            'Select the edge label between "{src}" and "{tgt}".',
        ]).format(src=s_text, tgt=o_text)
        return self._build_sample(
            graph=graph, level="L2", qtype="relation_edge_label_hidden_abstain_mcq",
            question=question, options=options, answer="cannot_be_determined",
            answer_text="cannot_be_determined", correct_option_indices=[],
            answer_type="single_choice",
            metadata={"is_unanswerable": True, "unanswerable_reason": "no_edge_label",
                      "used_disambiguation": bool(s_disamb or o_disamb)},
        )

    # =========================================================
    # Multihop (unchanged)
    # =========================================================

    def generate_relation_multihop_multiselect(self, graph: DiagramGraph,
                                               min_hops: int = 2, max_hops: int = 4) -> Optional[Dict[str, Any]]:
        usable_ids = self._usable_entity_ids(graph)
        top_level_ids = set(self._top_level_usable_entity_ids(graph))

        # Pass 1: top-level scope
        top_level_candidates = self._all_unique_multihop_pairs_in_scope(
            graph=graph, allowed_node_ids=top_level_ids,
            min_hops=min_hops, max_hops=max_hops,
        )
        filtered_top = []
        for src, tgt, path in top_level_candidates:
            if self._can_build_multihop_options_for_scope(graph, path, top_level_ids):
                filtered_top.append((src, tgt, path))

        if filtered_top:
            for src_id, tgt_id, path in sorted(filtered_top, key=lambda x: len(x[2]), reverse=True):
                result = self._build_and_emit_multihop_sample(
                    graph=graph, src_id=src_id, tgt_id=tgt_id, path=path,
                    scope_ids=top_level_ids, path_scope="top_level_only",
                )
                if result is not None:
                    return result

        # Pass 2: local-container scope
        candidate_containers = unique_preserve_order(
            [str(cid) for eid in usable_ids for cid in self._get_containers_of(graph, eid)]
        )
        self.rng.shuffle(candidate_containers)

        local_candidates = []
        for container_id in candidate_containers:
            scope_ids = set(self._usable_entities_in_container(graph, container_id))
            if len(scope_ids) < 3:
                continue
            for src_id, tgt_id, path in self._all_unique_multihop_pairs_in_scope(
                graph=graph, allowed_node_ids=scope_ids,
                min_hops=min_hops, max_hops=max_hops,
            ):
                shared = self._shared_container_for_path(graph, path)
                if (shared is not None and str(shared) == str(container_id)
                        and self._can_build_multihop_options_for_scope(graph, path, scope_ids)):
                    local_candidates.append((container_id, src_id, tgt_id, path, scope_ids))

        if local_candidates:
            for container_id, src_id, tgt_id, path, scope_ids in sorted(
                local_candidates, key=lambda x: len(x[3]), reverse=True
            ):
                container_display, _ = self._entity_display_text_and_flag(graph, container_id)
                if container_display is None:
                    continue
                result = self._build_and_emit_multihop_sample(
                    graph=graph, src_id=src_id, tgt_id=tgt_id, path=path,
                    scope_ids=scope_ids, path_scope="local_container",
                    container_display=container_display, container_id=container_id,
                )
                if result is not None:
                    return result

        return None

    # =========================================================
    # Conflict (unchanged)
    # =========================================================

    def generate_relation_conflict_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        if not graph.get_conflict_relations():
            return None
        relation_texts, conflict_texts = [], []
        for rel in self._all_relations(graph):
            s = self._display_text_or_none(graph, str(rel["subject_id"]))
            o = self._display_text_or_none(graph, str(rel["object_id"]))
            if s is None or o is None:
                continue
            text = f'{s} {rel.get("predicate", "")} {o}'
            relation_texts.append(text)
            if rel.get("is_conflict", False):
                conflict_texts.append(text)
        relation_texts = unique_preserve_order(relation_texts)
        conflict_texts = unique_preserve_order(conflict_texts)
        if not conflict_texts:
            return None
        correct = self.rng.choice(conflict_texts)
        distractors = [x for x in relation_texts if norm_key(x) != norm_key(correct)]
        if len(distractors) < 3:
            return None
        self.rng.shuffle(distractors)
        options, correct_indices, answer_labels = self._shuffle_options(
            [correct] + distractors[:3], [correct])
        if not self._validate_single_choice_options(options, correct_indices):
            return None
        question = self.rng.choice([
            "Which of the following relations is explicitly marked as conflicting?",
            "Select the relation that is annotated as a conflict.",
            "Which relation in the diagram is labeled as conflicting?",
        ])
        return self._build_sample(
            graph=graph, level="L3", qtype="relation_conflict_mcq", question=question,
            options=options, answer=answer_labels[0], answer_text=correct,
            correct_option_indices=correct_indices, answer_type="single_choice", metadata=None,
        )

    # =========================================================
    # Function summary (unchanged)
    # =========================================================

    def generate_function_summary_open(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        summary = normalize_ws(getattr(graph, "summary_text", "") or "")
        if not summary:
            return None
        question = self.rng.choice([
            "Briefly summarize the overall function or purpose of the diagram.",
            "What does this diagram describe at a high level?",
            "Provide a short summary of the system or process shown in the figure.",
        ])
        return self._build_sample(
            graph=graph, level="L3", qtype="function_summary_open", question=question,
            options=[], answer=summary, answer_text=summary, answer_type="open", metadata=None,
        )

    # =========================================================
    # Backward compatibility aliases
    # =========================================================

    def generate_relation_successor_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        return self.generate_relation_successor_hidden_abstain_mcq(graph)

    def generate_relation_containment_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        return self.generate_relation_containment_hidden_abstain_mcq(graph)

    def generate_relation_edge_label_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        return self.generate_relation_edge_label_hidden_abstain_mcq(graph)

    def generate_relation_unanswerable_successor_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        old = self.hidden_abstain_unanswerable_ratio
        try:
            self.hidden_abstain_unanswerable_ratio = 1.0
            return self.generate_relation_successor_hidden_abstain_mcq(graph)
        finally:
            self.hidden_abstain_unanswerable_ratio = old

    def generate_relation_unanswerable_containment_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        old = self.hidden_abstain_unanswerable_ratio
        try:
            self.hidden_abstain_unanswerable_ratio = 1.0
            return self.generate_relation_containment_hidden_abstain_mcq(graph)
        finally:
            self.hidden_abstain_unanswerable_ratio = old

    def generate_relation_unanswerable_edge_label_mcq(self, graph: DiagramGraph) -> Optional[Dict[str, Any]]:
        old = self.hidden_abstain_unanswerable_ratio
        try:
            self.hidden_abstain_unanswerable_ratio = 1.0
            return self.generate_relation_edge_label_hidden_abstain_mcq(graph)
        finally:
            self.hidden_abstain_unanswerable_ratio = old

    # =========================================================
    # Main entry point — [PLAN C] added 2 new question types
    # =========================================================

    def generate_for_graph(
            self,
            graph: DiagramGraph,
            enabled_question_types: Optional[List[str]] = None,
            max_per_type: int = 1,
    ) -> List[Dict[str, Any]]:
        qtype_to_func = {
            "element_presence_mcq": self.generate_element_presence_mcq,
            "element_absence_mcq": self.generate_element_absence_mcq,
            "relation_successor_hidden_abstain_mcq": self.generate_relation_successor_hidden_abstain_mcq,
            "relation_successor_multiselect": self.generate_relation_successor_multiselect,            # [PLAN C] NEW
            "relation_containment_hidden_abstain_mcq": self.generate_relation_containment_hidden_abstain_mcq,
            "relation_containment_multiselect": self.generate_relation_containment_multiselect,        # [PLAN C] NEW
            "relation_edge_label_hidden_abstain_mcq": self.generate_relation_edge_label_hidden_abstain_mcq,
            "relation_multihop_multiselect": self.generate_relation_multihop_multiselect,
            "relation_conflict_mcq": self.generate_relation_conflict_mcq,
            "function_summary_open": self.generate_function_summary_open,
            # backward compat aliases
            "relation_successor_mcq": self.generate_relation_successor_mcq,
            "relation_containment_mcq": self.generate_relation_containment_mcq,
            "relation_edge_label_mcq": self.generate_relation_edge_label_mcq,
            "relation_unanswerable_successor_mcq": self.generate_relation_unanswerable_successor_mcq,
            "relation_unanswerable_containment_mcq": self.generate_relation_unanswerable_containment_mcq,
            "relation_unanswerable_edge_label_mcq": self.generate_relation_unanswerable_edge_label_mcq,
        }
        enabled = enabled_question_types or list(qtype_to_func.keys())
        samples = []
        for qtype in enabled:
            if qtype not in qtype_to_func:
                continue
            gen_fn = qtype_to_func[qtype]
            seen_questions: Set[str] = set()
            tries = 0
            made = 0
            while made < max_per_type and tries < max(8, 6 * max_per_type):
                tries += 1
                sample = gen_fn(graph)
                if sample is None:
                    continue
                qk = norm_key(sample.get("question", ""))
                if qk in seen_questions:
                    continue
                seen_questions.add(qk)
                samples.append(sample)
                made += 1
        return samples