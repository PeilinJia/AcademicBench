"""
AcademicBench Evaluation Runner — Production Version

Scheme D: neutral prompt, no format/abstention hints, 3-stage post-hoc parsing.

Features absorbed from teacher's reference code:
  - Concurrent execution (ThreadPoolExecutor, configurable workers)
  - Image preprocessing (PIL thumbnail to 1024px + JPEG 85%)
  - Token usage tracking per sample and aggregate
  - Exponential backoff retry with jitter
  - Resumable JSONL cache with content-hash keys
  - Pipeline mode: run all models sequentially, single command

Usage:
    python run_evaluation.py \
        --benchmark benchmark_dataset.json \
        --image_dir data/images \
        --models gpt-4o gpt-4o-mini gemini-2.0-flash claude-sonnet-4-5 gemini-1.5-flash \
        --judge_model gpt-4o-mini \
        --out results \
        --workers 4 \
        --sleep 1.0
"""



import argparse, base64, hashlib, json, os, re, time, random, threading, io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple



try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# =====================================================================
# Prompts — NEUTRAL (Scheme D)
# =====================================================================
#用户消息，模型看到的是一道题配几个选项

NEUTRAL_SYSTEM = (
    "You are answering questions about an academic framework diagram. "
    "Base your answer on what is visible in the image."
)

MCQ_USER_TEMPLATE = "Question: {question}\n\nOptions:\n{options_block}"
OPEN_USER_TEMPLATE = "Question: {question}"

PARSE_JUDGE_SYSTEM = (
    "You classify how a model responded to a multiple-choice question "
    "about a diagram. You do NOT judge correctness — only what the "
    "model's response expresses."
)

PARSE_JUDGE_USER = """A model was shown a diagram and asked:

QUESTION: {question}

OPTIONS:
{options_block}

The model replied:
\"\"\"
{raw_output}
\"\"\"

Classify into one of:
  - "selected": chose one of the listed options (A/B/C/D).
  - "abstained": refused, said cannot determine, said none fit, or
                 expressed genuine uncertainty.
  - "invalid": empty, off-topic, or doesn't address the question.

If "selected", report the chosen letter.

Respond ONLY with JSON on one line:
{{"action": "selected"|"abstained"|"invalid", "letter": "A"|"B"|"C"|"D"|null}}"""

PARSE_JUDGE_MULTI_SYSTEM = (
    "You classify how a model responded to a SELECT-ALL-THAT-APPLY question "
    "about a diagram. You do NOT judge correctness — only which options the "
    "model's response actually endorses."
)

PARSE_JUDGE_MULTI_USER = """A model was shown a diagram and asked a SELECT-ALL-THAT-APPLY question:

QUESTION: {question}

OPTIONS:
{options_block}

The model replied:
\"\"\"
{raw_output}
\"\"\"

Decide which options the model ENDORSES as correct. Read carefully:
  - Include an option only if the model affirms it (says "Yes", "points to this",
    lists it among its answers, marks it correct).
  - EXCLUDE any option the model rejects (says "No", "not directly", "is an output",
    "outside the box", "does not apply").
  - If the model says none apply / cannot be determined / refuses -> "abstained".
  - If the reply is empty or off-topic -> "invalid".

Respond ONLY with JSON on one line:
{{"action": "selected"|"abstained"|"invalid", "letters": ["A","B"]}}"""

OPEN_JUDGE_SYSTEM = (
    "You are a strict scientific evaluator scoring a model's description "
    "of an academic framework diagram against a reference from the paper."
)

OPEN_JUDGE_USER = """Reference (from paper):
\"\"\"
{reference}
\"\"\"

Model's description:
\"\"\"
{prediction}
\"\"\"

Score on three axes, each 0-2:
1. COVERAGE: mentions key stages/components/roles?
2. CORRECTNESS: factually consistent with reference?
3. HALLUCINATION (higher=less hallucination): avoids inventing content?

Respond ONLY with JSON on one line:
{{"coverage": <int>, "correctness": <int>, "hallucination": <int>, "rationale": "<one sentence>"}}"""


# =====================================================================
# Image preprocessing
# =====================================================================

MAX_IMAGE_PX = 1024
JPEG_QUALITY = 85

#用PIL打开图片,按比例缩放到长边不超过1024像素,然后转成JPEG格式、85%质量,最后base64编码三个原因。
# 第一,学术论文截图分辨率经常在2000像素以上,直接传原图的base64非常大,会撞API的payload限制。
# 第二,多模态模型的图片token计费按图片大小来的,一张大图可能消耗1500个token,缩到1024像素大约700个token,成本砍一半。
# 第三,所有模型接收到完全一致的输入,保证对比的公平性。
def encode_image(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if HAS_PIL:
        with PILImage.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    else:
        mime = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg"}.get(ext, "image/png")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode(), mime


# =====================================================================
# Retry with exponential backoff + jitter
# =====================================================================

RETRIABLE_FRAGMENTS = (
    "ratelimit", "overloaded", "apiconnection", "timeout",
    "serviceunavailable", "internalserver",
)
RETRIABLE_STATUSES = {408, 429, 500, 502, 503, 504, 529}

def call_with_retry(fn, *args, max_retries=6, base_delay=4.0, **kwargs):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            cls = type(e).__name__.lower()
            status = getattr(e, "status_code",
                             getattr(getattr(e, "response", None), "status_code", None))
            retriable = (any(f in cls for f in RETRIABLE_FRAGMENTS)
                         or (isinstance(status, int) and status in RETRIABLE_STATUSES))
            if not retriable or attempt == max_retries:
                raise
            delay = min(60, base_delay * (2 ** attempt)) * (0.7 + 0.6 * random.random())
            print(f"  [retry] {type(e).__name__} attempt {attempt+1}/{max_retries}, "
                  f"wait {delay:.1f}s")
            time.sleep(delay)
    raise last_exc


# =====================================================================
# Backends
# =====================================================================
#抽象了一个统一的Backend接口
class BaseBackend:
    name = "base"
    def chat_with_image(self, system, user, b64, mime): raise NotImplementedError
    def chat_text(self, system, user): raise NotImplementedError

class OpenAIBackend(BaseBackend):
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI(); self.model = model; self.name = model
    def chat_with_image(self, system, user, b64, mime):
        content = [{"type": "text", "text": user}]
        if b64:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}})
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            max_tokens=500, temperature=0.0)
        u = r.usage
        return (r.choices[0].message.content or "",
                {"prompt_tokens": u.prompt_tokens if u else 0,
                 "completion_tokens": u.completion_tokens if u else 0})
    def chat_text(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=300, temperature=0.0)
        return r.choices[0].message.content or ""

class AnthropicBackend(BaseBackend):
    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic(); self.model = model; self.name = model
    def chat_with_image(self, system, user, b64, mime):
        content = []
        if b64:
            content.append({"type": "image",
                             "source": {"type": "base64", "media_type": mime, "data": b64}})
        content.append({"type": "text", "text": user})
        r = self.client.messages.create(
            model=self.model, max_tokens=500, system=system,
            messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        return (text,
                {"prompt_tokens": getattr(u, "input_tokens", 0),
                 "completion_tokens": getattr(u, "output_tokens", 0)})
    def chat_text(self, system, user):
        r = self.client.messages.create(
            model=self.model, max_tokens=300, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")

class GeminiBackend(BaseBackend):
    def __init__(self, model):
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.model_name = model; self.name = model
    def chat_with_image(self, system, user, b64, mime):
        from google.genai import types
        parts = [types.Part.from_text(text=system + "\n\n" + user)]
        if b64:
            parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
        r = self.client.models.generate_content(
            model=self.model_name, contents=parts,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=500))
        u = getattr(r, "usage_metadata", None)
        return (r.text or "",
                {"prompt_tokens": getattr(u, "prompt_token_count", 0),
                 "completion_tokens": getattr(u, "candidates_token_count", 0)})
    def chat_text(self, system, user):
        from google.genai import types
        r = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Part.from_text(text=system + "\n\n" + user)],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=300))
        return r.text or ""

class DashScopeBackend(BaseBackend):
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model; self.name = model
    def chat_with_image(self, system, user, b64, mime):
        content = [{"type": "text", "text": user}]
        if b64:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}})
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            max_tokens=500, temperature=0.0)
        u = r.usage
        return (r.choices[0].message.content or "",
                {"prompt_tokens": u.prompt_tokens if u else 0,
                 "completion_tokens": u.completion_tokens if u else 0})
    def chat_text(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=300, temperature=0.0)
        return r.choices[0].message.content or ""

class OpenRouterBackend(BaseBackend):
    """OpenAI 兼容；展示名(self.name)与真实 model id(self.model)解耦。
    self.name 进缓存键和 results_detail 的 model 字段，必须与 metrics 里已有命名一致。"""
    def __init__(self, display_name, model_id):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://academicbench.local",
                             "X-Title": "AcademicBench"},
        )
        self.model = model_id        # 真实调用 id，如 openai/gpt-4o
        self.name = display_name     # 展示名，如 gpt-4o
    def chat_with_image(self, system, user, b64, mime):
        content = [{"type": "text", "text": user}]
        if b64:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}})
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            max_tokens=500, temperature=0.0)
        u = r.usage
        return (r.choices[0].message.content or "",
                {"prompt_tokens": u.prompt_tokens if u else 0,
                 "completion_tokens": u.completion_tokens if u else 0})
    def chat_text(self, system, user):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=300, temperature=0.0)
        return r.choices[0].message.content or ""


# 展示名 -> (provider, 真实 model id)。展示名要与 metrics.json/缓存里已有的保持一致。
# OpenRouter 的 slug 用前请到 openrouter.ai/models（按 Vision 过滤）核对版本号。
MODEL_REGISTRY = {
    "gpt-4o":           ("openrouter", "openai/gpt-4o"),
    "gpt-4o-mini":      ("openrouter", "openai/gpt-4o-mini"),
    "gemini-2.5-flash": ("openrouter", "google/gemini-2.5-flash"),
    "gemini-2.5-pro":   ("openrouter", "google/gemini-2.5-pro"),
    "claude-sonnet-4.6": ("openrouter", "anthropic/claude-sonnet-4.6"),
    "llama-3.2-11b-vision": ("openrouter", "meta-llama/llama-3.2-11b-vision-instruct"),
    "gemma-3-27b":      ("openrouter", "google/gemma-3-27b-it"),
    "qwen2.5-vl-72b":   ("openrouter", "qwen/qwen2.5-vl-72b-instruct"),
    # 千问 Max/Plus 是阿里闭源，OpenRouter 上没有，走 DashScope
    "qwen-vl-max":      ("dashscope",  "qwen-vl-max"),
    "qwen-vl-plus":     ("dashscope",  "qwen-vl-plus"),
}

def make_backend(model: str) -> BaseBackend:
    if model in MODEL_REGISTRY:
        provider, model_id = MODEL_REGISTRY[model]
        if provider == "openrouter":
            return OpenRouterBackend(model, model_id)
        if provider == "dashscope":
            b = DashScopeBackend(model_id); b.name = model; return b
        if provider == "openai":
            b = OpenAIBackend(model_id); b.name = model; return b
    # 回退：未注册的名字仍按原前缀逻辑直连（保留原有能力）
    m = model.lower()
    if m.startswith("gpt") or m.startswith("o"): return OpenAIBackend(model)
    if m.startswith("claude"): return AnthropicBackend(model)
    if m.startswith("gemini"): return GeminiBackend(model)
    if m.startswith("qwen"): return DashScopeBackend(model)
    raise ValueError(f"Unknown model family: {model}")


# =====================================================================
# Prompt building
# =====================================================================

def build_prompt(sample):
    if sample["answer_type"] == "open":
        return OPEN_USER_TEMPLATE.format(question=sample["question"])
    lines = [f'{o["label"]}. {o["text"]}' for o in sample["options"]]
    return MCQ_USER_TEMPLATE.format(question=sample["question"],
                                     options_block="\n".join(lines))


# =====================================================================
# 3-stage MCQ parsing
# =====================================================================

def _norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

#三个匹配规则按优先级排列。先找'Answer: B'这种明确的回答格式;再找只有一个字母的独立行;
# 最后如果整段文本里只出现了一个独立的选项字母,也用它。注意最后一条的前提是'唯一出现'——如果文本里同时出现了A和B(比如'Not A, but B'),这条规则不触发,交给后面的阶段处理。
def stage1_regex(raw, valid="ABCD"):
    """只认显式答案标记，绝不再把英文冠词 a 当成 A。"""
    if not raw: return None
    t = raw.strip()
    # 1) "the answer is X" / "correct option is X" / "Answer: X"（取最后一次，结论通常在末尾）
    ms = list(re.finditer(
        rf"(?:correct\s+)?(?:answer|option|choice)\s*(?:is|are|:|=|would\s+be)\s*\(?\*{{0,2}}([{valid}])\b",
        t, re.I))
    if ms: return ms[-1].group(1).upper()
    # 2) 整个回复就是一个字母，如 "B" 或 "B."
    m = re.match(rf"^\W*([{valid}])\W*$", t, re.I)
    if m: return m.group(1).upper()
    # 3) 带选项标点的字母（"A." "B)" "(C)" "**D**"），且全文只出现一个才采用
    marks = re.findall(rf"(?<![A-Za-z])([{valid}])(?:\)|\*\*|[.\]:])", t)
    uniq = list(dict.fromkeys(x.upper() for x in marks))
    if len(uniq) == 1: return uniq[0]
    return None

def stage1_multi(raw, valid="ABCDEF"):
    """只抓“干净的显式结论列表”，如 'the correct options are A, B, and D'。
    若结论后面夹着其它文字（如把每个选项连同说明重列一遍），判为不干净，交给裁判。"""
    if not raw: return None
    anchors = list(re.finditer(
        r"(?:correct|right|final|selected)\s+(?:options?|answers?|choices?|modules?)\s*(?:are|is|:|=)|"
        r"(?:the\s+)?answers?\s*(?:are|is|:|=)",
        raw, re.I))
    for m in reversed(anchors):
        tail = re.split(r"[.\n;]", raw[m.end(): m.end()+90])[0]
        # 去掉字母/逗号/空白/and/&/括号/星号后若为空，说明是纯字母列表
        leftover = re.sub(rf"and|&|[{valid},\s()*]", "", tail, flags=re.I).strip()
        if leftover == "":
            letters = list(dict.fromkeys(re.findall(rf"[{valid}]", tail.upper())))
            if letters: return letters
    return None

_NEG = re.compile(r"\b(?:not|no|never|isn'?t|aren'?t|does\s*not|doesn'?t|don'?t|"
                  r"cannot|can'?t|outside|rather\s+than|instead\s+of|exclud)", re.I)

def stage2_sub(raw, options):
    """单选 substring：选项文字唯一命中且其前文无否定词时采用。"""
    if not raw: return None
    nr = _norm(raw)
    hits = []
    for o in options:
        ot = _norm(o["text"])
        if len(ot) < 3: continue
        pos = nr.find(ot)
        if pos < 0: continue
        if _NEG.search(nr[max(0, pos-40):pos]): continue
        hits.append(o["label"])
    return hits[0] if len(hits) == 1 else None

def parse_json_obj(text):
    if not text: return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

# 拒答措辞高精度识别：先于抓字母运行，挡住“说了没有/无法确定却被抓个字母”的情况
ABSTAIN_RE = re.compile(
    r"cannot be determined|can'?t be determined|cannot determine|"
    r"(?:does|do)\s*not\s*(?:directly\s*)?(?:point|lead|connect|map)\s*to\s*any|"
    r"none of the (?:above|options|listed|choices|modules|given|four)|"
    r"no module is (?:visually )?enclosed|nothing is (?:visually )?enclosed|"
    r"is a standalone (?:module|box|node|component)|"
    r"none (?:of these )?(?:fit|apply|are correct)|"
    r"no (?:such )?(?:arrow|module|connection|edge|option|stage) (?:exists|is present|is shown|is enclosed)",
    re.I)

def is_abstain(raw):
    return bool(raw) and bool(ABSTAIN_RE.search(raw))

def stage3_judge(judge_backend, sample, raw, multi=False):
    lines = [f'{o["label"]}. {o["text"]}' for o in sample["options"]]
    ob = "\n".join(lines)
    try:
        if multi:
            user = PARSE_JUDGE_MULTI_USER.format(question=sample["question"],
                                                  options_block=ob, raw_output=raw)
            out = call_with_retry(judge_backend.chat_text, PARSE_JUDGE_MULTI_SYSTEM, user)
        else:
            user = PARSE_JUDGE_USER.format(question=sample["question"],
                                            options_block=ob, raw_output=raw)
            out = call_with_retry(judge_backend.chat_text, PARSE_JUDGE_SYSTEM, user)
        obj = parse_json_obj(out) or {}
        action = obj.get("action", "invalid")
        if action not in {"selected", "abstained", "invalid"}: action = "invalid"
        if multi:
            letters = [str(x).upper() for x in (obj.get("letters") or [])
                       if str(x).upper() in "ABCDEF"]
            letters = list(dict.fromkeys(letters))
            if action == "selected" and not letters: action = "invalid"
            return {"action": action, "letter": None, "letters": letters}
        return {"action": action,
                "letter": obj.get("letter") if action == "selected" else None,
                "letters": None}
    except Exception:
        return {"action": "invalid", "letter": None, "letters": None}

@dataclass
class ParseResult:
    action: str; letter: Optional[str]; letters: Optional[List[str]]; stage: str

def parse_single(raw, sample, judge):
    if is_abstain(raw): return ParseResult("abstained", None, None, "abstain_rule")
    l = stage1_regex(raw, "ABCD")
    if l: return ParseResult("selected", l, None, "regex")
    l = stage2_sub(raw, sample["options"])
    if l: return ParseResult("selected", l, None, "substring")
    v = stage3_judge(judge, sample, raw, multi=False)
    return ParseResult(v["action"], v.get("letter"), None, "judge")

def parse_multi(raw, sample, judge):
    if is_abstain(raw): return ParseResult("abstained", None, None, "abstain_rule")
    ls = stage1_multi(raw, "ABCDEF")
    if ls: return ParseResult("selected", None, ls, "regex")
    v = stage3_judge(judge, sample, raw, multi=True)
    if v["action"] == "selected" and v.get("letters"):
        return ParseResult("selected", None, v["letters"], "judge")
    return ParseResult(v["action"], None, None, "judge")



# =====================================================================
# Scoring
# =====================================================================
#评分逻辑用一张2×3的矩阵来理解最清楚:行是题目类型(可回答/不可回答),列是模型行为(selected/abstained/invalid)。
#可回答的题:选对了1分,选错了0分,拒答了也是0分(假拒答)。不可回答的题:拒答了1分(正确拒答),选了任何选项都是0分(幻觉),invalid也是0分。
#这个评分矩阵直接决定了metrics里的四个核心指标:answerable_correct_rate(可回答题的答对率)、
# answerable_false_abstain_rate(可回答题的假拒答率)、unanswerable_correct_abstain_rate(不可回答题的正确拒答率)、unanswerable_hallucination_rate(不可回答题的幻觉率)。

def score_single(sample, pr):
    unans = bool(sample.get("metadata", {}).get("is_unanswerable"))
    if pr.action == "abstained": return 1.0 if unans else 0.0
    if pr.action == "invalid": return 0.0
    if unans: return 0.0
    return 1.0 if pr.letter == sample.get("answer") else 0.0

def score_multi(sample, pr):
    gold = set(sample.get("answer", []))
    if pr.action != "selected" or not pr.letters: return 0.0, 0.0
    pred = set(pr.letters)
    exact = 1.0 if pred == gold else 0.0
    if not pred and not gold: return 1.0, 1.0
    if not pred or not gold: return 0.0, 0.0
    tp = len(pred & gold); p = tp/len(pred); r = tp/len(gold)
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return exact, f1


# =====================================================================
# Cache
# =====================================================================

class Cache:
    def __init__(self, path):
        self.path = path; self.lock = threading.Lock(); self.data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self.data[obj["key"]] = obj
    def get(self, key): return self.data.get(key)
    def put(self, key, record):
        record = {"key": key, **record}
        with self.lock:
            self.data[key] = record
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

def sid_for(sample, idx):
    fp = json.dumps({"q": sample.get("question", ""),
                      "opts": [(o.get("label"), o.get("text")) for o in sample.get("options", [])],
                      "ans": sample.get("answer")}, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(fp.encode()).hexdigest()[:8]
    return f"{sample.get('image_id','unk')}::{idx}::{sample.get('question_type','unk')}::{h}"


# =====================================================================
# Per-sample worker
# =====================================================================

@dataclass
class ScoredResult:
    sample_id: str; model: str; level: str; question_type: str
    answer_type: str; is_unanswerable: bool; raw_output: str
    parse_action: Optional[str] = None; parse_letter: Optional[str] = None
    parse_letters: Optional[List[str]] = None; parse_stage: Optional[str] = None
    correct: Optional[float] = None; multi_f1: Optional[float] = None
    open_judge: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None; error: Optional[str] = None

def process_sample(idx, sample, backend, judge_backend, image_dir, cache, sleep):
    sid = sid_for(sample, idx)
    pred_key = f"pred::{backend.name}::{sid}"
    is_unans = bool(sample.get("metadata", {}).get("is_unanswerable"))
    res = ScoredResult(sample_id=sid, model=backend.name, level=sample["level"],
                        question_type=sample["question_type"],
                        answer_type=sample["answer_type"],
                        is_unanswerable=is_unans, raw_output="")

    cached = cache.get(pred_key)
    if cached and not cached.get("error"):
        raw = cached.get("raw_output", "")
        res.usage = cached.get("usage")
    else:
        image_path = os.path.join(image_dir, sample["image_id"])
        try:
            b64, mime = encode_image(image_path)
            user = build_prompt(sample)
            raw, usage = call_with_retry(backend.chat_with_image,
                                          NEUTRAL_SYSTEM, user, b64, mime)
            cache.put(pred_key, {"raw_output": raw, "usage": usage,
                                  "question_type": sample["question_type"]})
            res.usage = usage
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            cache.put(pred_key, {"raw_output": "", "error": err,
                                  "question_type": sample["question_type"]})
            res.error = err; return res
        if sleep: time.sleep(sleep)

    res.raw_output = raw

    if sample["answer_type"] == "single_choice":
        pk = f"parse2::{backend.name}::{sid}"
        cp = cache.get(pk)
        if cp:
            pr = ParseResult(cp["action"], cp.get("letter"), None, cp.get("stage", "cache"))
        else:
            pr = parse_single(raw, sample, judge_backend)
            cache.put(pk, {"action": pr.action, "letter": pr.letter, "stage": pr.stage})
        res.parse_action = pr.action; res.parse_letter = pr.letter; res.parse_stage = pr.stage
        res.correct = score_single(sample, pr)

    elif sample["answer_type"] == "multi_select":
        pk = f"parse2::{backend.name}::{sid}"
        cp = cache.get(pk)
        if cp:
            pr = ParseResult(cp["action"], None, cp.get("letters"), cp.get("stage", "cache"))
        else:
            pr = parse_multi(raw, sample, judge_backend)
            cache.put(pk, {"action": pr.action, "letters": pr.letters, "stage": pr.stage})
        res.parse_action = pr.action; res.parse_letters = pr.letters; res.parse_stage = pr.stage
        exact, f1 = score_multi(sample, pr)
        res.correct = exact; res.multi_f1 = f1

    else:
        jk = f"open_judge::{backend.name}::{sid}"
        cj = cache.get(jk)
        if cj:
            res.open_judge = cj.get("judge")
        else:
            user = OPEN_JUDGE_USER.format(reference=sample.get("answer_text", ""),
                                           prediction=raw)
            try:
                out = call_with_retry(judge_backend.chat_text, OPEN_JUDGE_SYSTEM, user)
                parsed = parse_json_obj(out)
                if parsed:
                    for k in ("coverage", "correctness", "hallucination"):
                        parsed[k] = int(parsed.get(k, 0))
                res.open_judge = parsed
                cache.put(jk, {"judge": parsed, "raw": out})
            except Exception as e:
                cache.put(jk, {"judge": None, "error": str(e)})
    return res


# =====================================================================
# Run one model
# =====================================================================

def run_model(model_name, samples, image_dir, cache, judge_backend, workers, sleep):
    backend = make_backend(model_name)
    results = []; total = len(samples)
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Samples: {total}  |  Workers: {workers}")
    print(f"{'='*60}")
    start = time.time()

    if workers <= 1:
        for idx, sample in enumerate(samples):
            r = process_sample(idx, sample, backend, judge_backend, image_dir, cache, sleep)
            results.append(r)
            if (idx+1) % 20 == 0 or idx == total-1:
                print(f"  [{model_name}] {idx+1}/{total}  elapsed={time.time()-start:.0f}s")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_sample, idx, s, backend, judge_backend,
                                    image_dir, cache, sleep): idx
                       for idx, s in enumerate(samples)}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result()); done += 1
                if done % 20 == 0 or done == total:
                    errs = sum(1 for x in results if x.error)
                    print(f"  [{model_name}] {done}/{total}  "
                          f"elapsed={time.time()-start:.0f}s  errors={errs}")

    elapsed = time.time() - start
    errs = sum(1 for x in results if x.error)
    m, s = divmod(int(elapsed), 60)
    print(f"  [{model_name}] DONE in {m}m{s}s  errors={errs}/{total}")
    return results


# =====================================================================
# Aggregation
# =====================================================================

def aggregate(results):
    by_model = defaultdict(lambda: {
        "ans_correct": [], "ans_false_abstain": [],
        "unans_correct_abstain": [], "unans_hallucination": [],
        "invalid": [], "by_level": defaultdict(list),
        "by_qtype": defaultdict(list), "multi_f1": [],
        "open_score": [], "stage_counts": defaultdict(int),
        "prompt_tokens": 0, "completion_tokens": 0,
    })
    for r in results:
        b = by_model[r.model]
        if r.usage:
            b["prompt_tokens"] += r.usage.get("prompt_tokens", 0)
            b["completion_tokens"] += r.usage.get("completion_tokens", 0)
        if r.error: b["invalid"].append(1.0); continue
        if r.answer_type == "open":
            if r.open_judge:
                s = (r.open_judge["coverage"] + r.open_judge["correctness"]
                      + r.open_judge["hallucination"]) / 6.0
                b["open_score"].append(s)
                b["by_level"][r.level].append(s)
                b["by_qtype"][r.question_type].append(s)
            continue
        if r.parse_stage: b["stage_counts"][r.parse_stage] += 1
        if r.parse_action == "invalid":
            b["invalid"].append(1.0)
            b["by_level"][r.level].append(0.0); b["by_qtype"][r.question_type].append(0.0)
            continue
        b["invalid"].append(0.0)   # FIX: 有效 MCQ 作答也计入分母，否则 invalid_response_rate 恒为 1.0
        # 多选用 F1(部分分)进入各层级/题型统计，与 multihop_f1 口径一致；
        # 单选仍是精确匹配。每条样本的 exact 值仍保存在 results_detail.json。
        score = r.multi_f1 if (r.answer_type == "multi_select" and r.multi_f1 is not None) \
                else (r.correct if r.correct is not None else 0.0)
        b["by_level"][r.level].append(score)
        b["by_qtype"][r.question_type].append(score)
        if r.is_unanswerable:
            if r.parse_action == "abstained":
                b["unans_correct_abstain"].append(1.0); b["unans_hallucination"].append(0.0)
            else:
                b["unans_correct_abstain"].append(0.0); b["unans_hallucination"].append(1.0)
        else:
            if r.parse_action == "abstained":
                b["ans_correct"].append(0.0); b["ans_false_abstain"].append(1.0)
            else:
                b["ans_correct"].append(score); b["ans_false_abstain"].append(0.0)
        if r.answer_type == "multi_select" and r.multi_f1 is not None:
            b["multi_f1"].append(r.multi_f1)

    def m(xs): return round(sum(xs)/len(xs), 4) if xs else None
    out = {}
    for model, b in by_model.items():
        out[model] = {
            "answerable_correct_rate": m(b["ans_correct"]),
            "answerable_false_abstain_rate": m(b["ans_false_abstain"]),
            "unanswerable_correct_abstain_rate": m(b["unans_correct_abstain"]),
            "unanswerable_hallucination_rate": m(b["unans_hallucination"]),
            "invalid_response_rate": m(b["invalid"]),
            "multihop_f1": m(b["multi_f1"]),
            "l3_open_score": m(b["open_score"]),
            "by_level": {k: m(v) for k, v in sorted(b["by_level"].items())},
            "by_qtype": {k: m(v) for k, v in sorted(b["by_qtype"].items())},
            "parse_stage_counts": dict(b["stage_counts"]),
            "token_usage": {
                "prompt_tokens": b["prompt_tokens"],
                "completion_tokens": b["completion_tokens"],
                "total_tokens": b["prompt_tokens"] + b["completion_tokens"],
            },
        }
    return out


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="AcademicBench Evaluation Runner")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 条样本（0=全部），用于廉价连通性测试")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.benchmark, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[:args.limit]
        print(f"  [LIMIT] 只跑前 {args.limit} 条样本（连通性测试模式）")

    print(f"\nAcademicBench Evaluation Pipeline")
    print(f"  Benchmark: {len(samples)} samples")
    print(f"  Models: {args.models}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Workers: {args.workers}  Sleep: {args.sleep}s")

    cache = Cache(os.path.join(args.out, "cache.jsonl"))
    judge = make_backend(args.judge_model)
    all_results = []

    for model in args.models:
        res = run_model(model, samples, args.image_dir, cache, judge,
                         workers=args.workers, sleep=args.sleep)
        all_results.extend(res)
        metrics = aggregate(all_results)
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  -> metrics.json updated ({model} done)")

    with open(os.path.join(args.out, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {args.out}/")

if __name__ == "__main__":
    main()