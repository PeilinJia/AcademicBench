# question_safety_validator.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def norm_key(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_preserve_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        k = norm_key(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


class QuestionSafetyValidator:
    """
    独立的题目安全校验器。

    调用方式：
        ok, reason = validator.validate(item, graph, pg=self)

    其中 pg 是 PromptGenerator 实例，用于复用：
        - _usable_entity_ids(graph)
        - _display_text_or_none(graph, eid)

    多跳题验证策略：
        优先使用 metadata["intermediate_node_ids"]（prompt_generator 写入的 ground truth）
        直接验证，完全不依赖反查路径函数（这些函数在 PromptGenerator 中不存在）。
    """

    def validate(
        self,
        item: Dict[str, Any],
        graph: Any,
        pg: Any = None,
    ) -> Tuple[bool, str]:
        ok, reason = self._validate_common(item)
        if not ok:
            return ok, reason

        qtype = item.get("question_type", "")

        if qtype == "element_presence_mcq":
            return self._validate_element_presence(item, graph, pg)

        if qtype == "element_absence_mcq":
            return self._validate_element_absence(item, graph, pg)

        if qtype == "relation_multihop_multiselect":
            return self._validate_relation_multihop(item, graph, pg)

        return True, "ok"

    # =========================
    # 通用校验
    # =========================

    def _validate_common(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        options = item.get("options", [])
        answer_type = item.get("answer_type")
        answer = item.get("answer", [])
        answer_text = item.get("answer_text", [])
        correct_option_indices = item.get("correct_option_indices", [])

        if not isinstance(options, list) or len(options) < 2:
            return False, "common: options missing or too few"

        option_texts = []
        option_labels = []
        for op in options:
            if not isinstance(op, dict):
                return False, "common: option is not dict"
            if "label" not in op or "text" not in op:
                return False, "common: option missing label/text"
            option_labels.append(op["label"])
            option_texts.append(op["text"])

        # 1) label 唯一
        if len(option_labels) != len(set(option_labels)):
            return False, "common: duplicate option labels"

        # 2) text 归一化后唯一
        norm_option_texts = [norm_key(x) for x in option_texts]
        if len(norm_option_texts) != len(set(norm_option_texts)):
            return False, "common: duplicate option texts after normalization"

        # 3) correct_option_indices 合法
        if not isinstance(correct_option_indices, list):
            return False, "common: correct_option_indices is not list"
        if any((not isinstance(i, int)) for i in correct_option_indices):
            return False, "common: non-int index in correct_option_indices"
        if any(i < 0 or i >= len(options) for i in correct_option_indices):
            return False, "common: correct_option_indices out of range"

        # -------------------------------------------------------------------
        # 4) unanswerable 题（correct_option_indices 为空）单独处理
        #    answer / answer_text 均为固定字符串 "cannot_be_determined"，
        #    无需与 options 对齐，直接放行。
        # -------------------------------------------------------------------
        if len(correct_option_indices) == 0:
            if answer != "cannot_be_determined":
                return False, "common: empty correct_option_indices but answer is not 'cannot_be_determined'"
            return True, "ok"

        # 5) answer labels 与 indices 一致
        expected_labels = [options[i]["label"] for i in correct_option_indices]
        if isinstance(answer, list):
            if sorted(answer) != sorted(expected_labels):
                return False, "common: answer labels mismatch correct_option_indices"
        else:
            if [answer] != expected_labels:
                return False, "common: answer label mismatch correct_option_indices"

        # 6) answer_text 与 indices 一致
        expected_texts = [options[i]["text"] for i in correct_option_indices]
        if answer_type == "multi_select":
            if not isinstance(answer_text, list):
                return False, "common: answer_text should be list for multi_select"
            if {norm_key(x) for x in answer_text} != {norm_key(x) for x in expected_texts}:
                return False, "common: answer_text mismatch correct_option_indices"
        else:
            # single_choice：answer_text 可以是 str 或单元素 list
            if isinstance(answer_text, list):
                if len(answer_text) != 1:
                    return False, "common: answer_text list length != 1 for single_choice"
                if norm_key(answer_text[0]) != norm_key(expected_texts[0]):
                    return False, "common: answer_text mismatch correct_option_indices"
            else:
                if len(expected_texts) == 0:
                    return False, "common: no expected texts for single_choice"
                if norm_key(answer_text) != norm_key(expected_texts[0]):
                    return False, "common: answer_text mismatch correct_option_indices"

        return True, "ok"

    # =========================
    # 图文本辅助
    # =========================

    def _graph_entity_texts(self, graph: Any, pg: Any) -> List[str]:
        if pg is None:
            return []
        if not hasattr(pg, "_usable_entity_ids") or not hasattr(pg, "_display_text_or_none"):
            return []
        texts = []
        for eid in pg._usable_entity_ids(graph):
            t = pg._display_text_or_none(graph, eid)
            if t:
                texts.append(t)
        return unique_preserve_order(texts)

    def _graph_entity_text_keys(self, graph: Any, pg: Any) -> Set[str]:
        return {norm_key(x) for x in self._graph_entity_texts(graph, pg)}

    def _extract_quoted_spans(self, question: str) -> List[str]:
        if not question:
            return []
        return re.findall(r'"([^"]+)"', question)

    # =========================
    # element_presence_mcq
    # =========================

    def _validate_element_presence(
        self,
        item: Dict[str, Any],
        graph: Any,
        pg: Any = None,
    ) -> Tuple[bool, str]:
        if pg is None:
            return False, "presence: pg is required"

        graph_keys = self._graph_entity_text_keys(graph, pg)
        options = item["options"]
        correct_idx = item["correct_option_indices"][0]
        correct_text = options[correct_idx]["text"]

        if norm_key(correct_text) not in graph_keys:
            return False, f'presence: correct option not in graph: "{correct_text}"'

        for i, op in enumerate(options):
            if i == correct_idx:
                continue
            if norm_key(op["text"]) in graph_keys:
                return False, f'presence: distractor unexpectedly exists in graph: "{op["text"]}"'

        return True, "ok"

    # =========================
    # element_absence_mcq
    # =========================

    def _validate_element_absence(
        self,
        item: Dict[str, Any],
        graph: Any,
        pg: Any = None,
    ) -> Tuple[bool, str]:
        if pg is None:
            return False, "absence: pg is required"

        graph_keys = self._graph_entity_text_keys(graph, pg)
        options = item["options"]
        correct_idx = item["correct_option_indices"][0]
        correct_text = options[correct_idx]["text"]

        if norm_key(correct_text) in graph_keys:
            return False, f'absence: correct option unexpectedly exists in graph: "{correct_text}"'

        for i, op in enumerate(options):
            if i == correct_idx:
                continue
            if norm_key(op["text"]) not in graph_keys:
                return False, f'absence: distractor not found in graph: "{op["text"]}"'

        return True, "ok"

    # =========================
    # relation_multihop_multiselect
    # =========================

    def _validate_relation_multihop(
        self,
        item: Dict[str, Any],
        graph: Any,
        pg: Any = None,
    ) -> Tuple[bool, str]:
        if pg is None:
            return False, "multihop: pg is required"

        metadata = item.get("metadata") or {}
        options = item.get("options", [])
        correct_option_indices = item.get("correct_option_indices", [])
        answer_text = item.get("answer_text")
        answer_type = item.get("answer_type", "")

        # ------------------------------------------------------------------
        # Ground truth source: metadata["intermediate_node_ids"]
        #
        # The prompt_generator always writes this field with the exact entity
        # IDs of the intermediate nodes on the verified unique path.
        # We resolve their current display texts here and use those as the
        # authoritative gold set — no reverse-lookup of answer_text strings
        # is needed or attempted.
        #
        # This avoids the "resolved=[]" failure that occurred when the old
        # validator tried to find entity IDs by matching display strings
        # (which can be disambiguated composite phrases like
        # "X immediately after Y") against raw entity texts in the graph.
        # ------------------------------------------------------------------
        intermediate_node_ids = metadata.get("intermediate_node_ids")

        if intermediate_node_ids is not None:
            # Resolve display texts from ground-truth IDs
            gold_displays = []
            for eid in intermediate_node_ids:
                disp = pg._display_text_or_none(graph, eid)
                if disp is None:
                    return False, f"multihop: intermediate node {eid} has no display text"
                gold_displays.append(disp)
            gold_displays = unique_preserve_order(gold_displays)
            gold_keys = {norm_key(x) for x in gold_displays}

            # 1) answer_text must match gold (type-aware)
            if answer_type == "multi_select":
                if not isinstance(answer_text, list):
                    return False, "multihop: answer_text should be list for multi_select"
                at_keys = {norm_key(x) for x in answer_text}
            else:
                # single_choice multihop: answer_text is a plain string
                if isinstance(answer_text, list):
                    if len(answer_text) != 1:
                        return False, "multihop: single_choice answer_text list length != 1"
                    at_keys = {norm_key(answer_text[0])}
                else:
                    at_keys = {norm_key(answer_text)}

            if at_keys != gold_keys:
                return False, (
                    f"multihop: answer_text does not match resolved intermediates; "
                    f"gold={sorted(gold_keys)}, answer_text={sorted(at_keys)}"
                )

            # 2) correct_option_indices must point exactly to gold options
            correct_texts_from_indices = {
                norm_key(options[i]["text"])
                for i in correct_option_indices
                if 0 <= i < len(options)
            }
            if correct_texts_from_indices != gold_keys:
                return False, (
                    f"multihop: correct_option_indices do not align with gold intermediates; "
                    f"gold={sorted(gold_keys)}, from_indices={sorted(correct_texts_from_indices)}"
                )

            # 3) distractors must not overlap gold
            for i, op in enumerate(options):
                if i in correct_option_indices:
                    continue
                if norm_key(op["text"]) in gold_keys:
                    return False, f'multihop: distractor is a true intermediate: "{op["text"]}"'

            # 4) src / tgt must not appear in gold
            question = item.get("question", "")
            quoted = self._extract_quoted_spans(question)
            if len(quoted) >= 2:
                st_keys = {norm_key(quoted[0]), norm_key(quoted[1])}
                if gold_keys & st_keys:
                    return False, "multihop: source/target included in answer_text"

            return True, "ok"

        # ------------------------------------------------------------------
        # Fallback: metadata["intermediate_node_ids"] is absent.
        # This should never happen with the current prompt_generator, but
        # we keep a conservative check for backward compatibility.
        # ------------------------------------------------------------------
        gold_keys_fb: Set[str]
        if answer_type == "multi_select":
            if not isinstance(answer_text, list):
                return False, "multihop(fb): answer_text should be list for multi_select"
            gold_keys_fb = {norm_key(x) for x in answer_text}
        else:
            if isinstance(answer_text, list):
                gold_keys_fb = {norm_key(x) for x in answer_text}
            else:
                gold_keys_fb = {norm_key(answer_text)} if answer_text else set()

        # correct indices must align with answer_text
        for i in correct_option_indices:
            if i < 0 or i >= len(options):
                return False, "multihop(fb): correct index out of range"
            if norm_key(options[i]["text"]) not in gold_keys_fb:
                return False, "multihop(fb): correct_option_indices not aligned with answer_text"

        # distractors must not overlap
        for i, op in enumerate(options):
            if i in correct_option_indices:
                continue
            if norm_key(op["text"]) in gold_keys_fb:
                return False, f'multihop(fb): distractor overlaps gold: "{op["text"]}"'

        question = item.get("question", "")
        quoted = self._extract_quoted_spans(question)
        if len(quoted) >= 2:
            st_keys = {norm_key(quoted[0]), norm_key(quoted[1])}
            if gold_keys_fb & st_keys:
                return False, "multihop(fb): source/target leaked into answer_text"

        return True, "ok_fallback"
