import os
import json
import base64
import pandas as pd
import streamlit as st

ANNOTATION_DIR = "./data/annotations"
IMAGE_DIR = "./data/images"

os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="AcademicBench Annotation Tool")

ENTITY_TYPES = ["Node", "Container"]
PREDICATES = ["flows_to", "contains", "connects_to"]


def default_data(img_filename):
    return {
        "image_id": img_filename,
        "paper_id": os.path.splitext(img_filename)[0],
        "summary": {
            "text": ""
        },
        "entity_editor_df": pd.DataFrame(columns=["text", "type"]),
        "entities": pd.DataFrame(columns=["id", "text", "type"]),
        "relations": pd.DataFrame(columns=["subject_id", "predicate", "object_id", "edge_text", "is_conflict"]),
        "_load_warning": ""
    }


def ensure_entity_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    实体编辑器只让用户编辑 text / type，
    id 自动生成，避免手工填错。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["text", "type"])

    df = df.copy()

    if "text" not in df.columns:
        df["text"] = ""
    if "type" not in df.columns:
        df["type"] = "Node"

    df["text"] = df["text"].astype(str)
    df["type"] = df["type"].astype(str)
    df["type"] = df["type"].apply(lambda x: x if x in ENTITY_TYPES else "Node")

    return df[["text", "type"]]


def build_entities_with_auto_ids(entity_editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 text/type 自动生成 e0, e1, e2...
    空文本行自动删除。
    """
    df = ensure_entity_editor_df(entity_editor_df).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()

    df = df[df["text"] != ""].reset_index(drop=True)
    df["type"] = df["type"].apply(lambda x: x if x in ENTITY_TYPES else "Node")

    df.insert(0, "id", [f"e{i}" for i in range(len(df))])
    return df[["id", "text", "type"]]


def ensure_relation_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["subject_id", "predicate", "object_id", "edge_text", "is_conflict"])

    df = df.copy()

    for col in ["subject_id", "predicate", "object_id", "edge_text", "is_conflict"]:
        if col not in df.columns:
            df[col] = False if col == "is_conflict" else ""

    df["subject_id"] = df["subject_id"].astype(str)
    df["predicate"] = df["predicate"].astype(str)
    df["object_id"] = df["object_id"].astype(str)
    df["edge_text"] = df["edge_text"].astype(str)
    df["is_conflict"] = df["is_conflict"].astype(bool)

    df["predicate"] = df["predicate"].apply(lambda x: x if x in PREDICATES else "flows_to")

    return df[["subject_id", "predicate", "object_id", "edge_text", "is_conflict"]]


def try_load_paper_id(data, img_filename):
    if data.get("paper_id"):
        return str(data["paper_id"]).strip()

    if isinstance(data.get("paper_info"), dict):
        pid = str(data["paper_info"].get("paper_id", "")).strip()
        if pid:
            return pid

    return os.path.splitext(img_filename)[0]


def try_load_summary(data):
    """
    兼容多种旧格式：
    1) 新格式:
       "summary": {"text": "..."}
    2) 老格式:
       "summary": "..."
    3) 更旧格式:
       "summary_annotation": {"functional_summary": "..."}
    """
    if isinstance(data.get("summary"), dict):
        return {"text": str(data["summary"].get("text", "")).strip()}

    if isinstance(data.get("summary"), str):
        return {"text": str(data.get("summary", "")).strip()}

    if isinstance(data.get("summary_annotation"), dict):
        return {"text": str(data["summary_annotation"].get("functional_summary", "")).strip()}

    return {"text": ""}


def load_annotation(img_filename):
    filename_base = os.path.splitext(img_filename)[0]
    json_path = os.path.join(ANNOTATION_DIR, f"{filename_base}.json")
    default = default_data(img_filename)

    if not os.path.exists(json_path):
        return default

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paper_id = try_load_paper_id(data, img_filename)
    summary = try_load_summary(data)

    raw_entities = data.get("entities", [])
    ent_df = pd.DataFrame(raw_entities) if raw_entities else pd.DataFrame(columns=["id", "text", "type"])

    if not ent_df.empty:
        if "text" not in ent_df.columns:
            ent_df["text"] = ""
        if "type" not in ent_df.columns:
            ent_df["type"] = "Node"

        ent_df["type"] = ent_df["type"].astype(str).replace({
            "Module/Block": "Node",
            "Text/Label": "Node"
        })
        ent_df["type"] = ent_df["type"].apply(lambda x: x if x in ENTITY_TYPES else "Node")

    entity_editor_df = ensure_entity_editor_df(ent_df)
    normalized_entities = build_entities_with_auto_ids(entity_editor_df)

    raw_relations = data.get("relations", [])
    relation_rows = []
    for r in raw_relations:
        subject_id = r.get("subject_id", "")
        object_id = r.get("object_id", "")

        if subject_id == "" and "subject" in r:
            subject_id = r.get("subject", "")
        if object_id == "" and "object" in r:
            object_id = r.get("object", "")

        relation_rows.append({
            "subject_id": str(subject_id).strip(),
            "predicate": str(r.get("predicate", "flows_to")).strip() or "flows_to",
            "object_id": str(object_id).strip(),
            "edge_text": str(r.get("edge_text", "")).strip(),
            "is_conflict": bool(r.get("is_conflict", False))
        })

    rel_df = ensure_relation_df(pd.DataFrame(relation_rows) if relation_rows else None)

    load_warning = ""
    raw_ids = []
    if not ent_df.empty and "id" in ent_df.columns:
        raw_ids = [str(x).strip() for x in ent_df["id"].tolist()]

    expected_ids = [f"e{i}" for i in range(len(normalized_entities))]
    raw_id_valid = (raw_ids == expected_ids)

    valid_ids = set(normalized_entities["id"].tolist())
    if not rel_df.empty:
        if raw_id_valid:
            rel_df = rel_df[
                rel_df["subject_id"].isin(valid_ids) &
                rel_df["object_id"].isin(valid_ids) &
                (rel_df["subject_id"] != rel_df["object_id"])
            ].reset_index(drop=True)
        else:
            rel_df = pd.DataFrame(columns=["subject_id", "predicate", "object_id", "edge_text", "is_conflict"])
            load_warning = (
                "检测到旧标注中的实体 ID 不是规范的 e0, e1, e2...，"
                "关系无法可靠继承。当前已自动重建实体 ID，请重新检查并标注关系。"
            )

    return {
        "image_id": data.get("image_id", img_filename),
        "paper_id": paper_id,
        "summary": summary,
        "entity_editor_df": entity_editor_df,
        "entities": normalized_entities,
        "relations": rel_df,
        "_load_warning": load_warning
    }


def save_annotation(img_filename, paper_id, summary_text, entity_editor_df, relations_df):
    filename_base = os.path.splitext(img_filename)[0]
    json_path = os.path.join(ANNOTATION_DIR, f"{filename_base}.json")

    entities_df = build_entities_with_auto_ids(entity_editor_df)
    relations_df = ensure_relation_df(relations_df)

    valid_ids = set(entities_df["id"].tolist())
    clean_relations = []

    if not relations_df.empty:
        for _, row in relations_df.iterrows():
            subj = str(row.get("subject_id", "")).strip()
            obj = str(row.get("object_id", "")).strip()
            pred = str(row.get("predicate", "flows_to")).strip() or "flows_to"
            edge_text = str(row.get("edge_text", "")).strip()
            is_conflict = bool(row.get("is_conflict", False))

            if subj not in valid_ids or obj not in valid_ids:
                continue
            if subj == obj:
                continue
            if pred not in PREDICATES:
                pred = "flows_to"

            clean_relations.append({
                "subject_id": subj,
                "predicate": pred,
                "object_id": obj,
                "edge_text": edge_text,
                "is_conflict": is_conflict
            })

    output = {
        "image_id": img_filename,
        "paper_id": str(paper_id).strip() if str(paper_id).strip() else filename_base,
        "summary": {
            "text": str(summary_text).strip()
        },
        "entities": entities_df.to_dict(orient="records"),
        "relations": clean_relations
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return True


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_zoomable_image(image_path: str, zoom_percent: int = 160, max_height: int = 900):
    """
    使用 HTML 显示图片，支持：
    - 更大的显示宽度
    - 横向滚动
    - 不再被列宽强制压缩得太小
    """
    if not os.path.exists(image_path):
        st.error(f"图片不存在: {image_path}")
        return

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".gif":
        mime = "image/gif"
    elif ext == ".svg":
        # SVG 不走 base64 img 更方便，但这里简单统一按文件文本处理也行
        try:
            with open(image_path, "r", encoding="utf-8") as f:
                svg_content = f.read()
            st.markdown(
                f"""
                <div style="
                    overflow:auto;
                    border:1px solid #ddd;
                    padding:8px;
                    background:#fafafa;
                    max-height:{max_height}px;
                ">
                    <div style="width:{zoom_percent}%;">
                        {svg_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            return
        except Exception:
            st.image(image_path, use_container_width=True)
            return

    b64 = image_to_base64(image_path)
    html = f"""
    <div style="
        overflow:auto;
        border:1px solid #ddd;
        padding:8px;
        background:#fafafa;
        max-height:{max_height}px;
    ">
        <img
            src="data:{mime};base64,{b64}"
            style="
                width:{zoom_percent}%;
                max-width:none;
                height:auto;
                display:block;
            "
        />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_image_panel(selected_file: str):
    image_path = os.path.join(IMAGE_DIR, selected_file)

    st.subheader("图像查看")
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.6])

    with col_a:
        zoom_percent = st.slider(
            "缩放比例 (%)",
            min_value=50,
            max_value=400,
            value=180,
            step=10,
            help="放大图片以便看清小字、箭头和容器结构。"
        )

    with col_b:
        max_height = st.slider(
            "查看区域高度",
            min_value=400,
            max_value=1400,
            value=950,
            step=50,
            help="控制图片查看框高度。图很长或很大时可以调高。"
        )

    with col_c:
        st.caption(
            "提示：放大后可在图片区域内横向/纵向滚动。"
            "复杂图推荐缩放到 180%~300%。"
        )

    render_zoomable_image(
        image_path=image_path,
        zoom_percent=zoom_percent,
        max_height=max_height
    )


def render_annotation_panel(anno, selected_file):
    if anno.get("_load_warning"):
        st.warning(anno["_load_warning"])

    st.subheader("1. Basic Info")
    paper_id = st.text_input(
        "Paper ID",
        value=str(anno.get("paper_id", "")),
        help="仅保留 paper_id 作为来源索引。"
    )

    st.divider()
    st.subheader("2. Functional Summary (L3)")
    summary_text = st.text_area(
        "Summary Text",
        value=anno["summary"].get("text", ""),
        height=220,
        help="填写来自论文原文/图注/方法描述整理出的整图功能概括，作为 L3 参考答案。"
    )

    st.divider()
    st.subheader("3. Entities (L1)")
    st.caption("只编辑 Text 和 Type；ID 自动按顺序生成：e0, e1, e2...")

    edited_entity_editor_df = st.data_editor(
        ensure_entity_editor_df(anno["entity_editor_df"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "text": st.column_config.TextColumn("Text", required=True),
            "type": st.column_config.SelectboxColumn("Type", options=ENTITY_TYPES, required=True),
        },
        key=f"entity_editor_{selected_file}"
    )

    generated_entities = build_entities_with_auto_ids(edited_entity_editor_df)

    st.caption("自动生成的实体 ID 预览")
    st.dataframe(generated_entities, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4. Relations (L2 + Robustness)")
    current_ids = generated_entities["id"].tolist()

    edited_relations = st.data_editor(
        ensure_relation_df(anno["relations"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "subject_id": st.column_config.SelectboxColumn("Start", options=current_ids, required=True),
            "predicate": st.column_config.SelectboxColumn("Predicate", options=PREDICATES, required=True),
            "object_id": st.column_config.SelectboxColumn("End", options=current_ids, required=True),
            "edge_text": st.column_config.TextColumn("Edge Text"),
            "is_conflict": st.column_config.CheckboxColumn("Conflict / Trap")
        },
        key=f"relation_editor_{selected_file}"
    )

    st.divider()
    if st.button("保存标注 JSON", type="primary", use_container_width=True):
        ok = save_annotation(
            img_filename=selected_file,
            paper_id=paper_id,
            summary_text=summary_text,
            entity_editor_df=edited_entity_editor_df,
            relations_df=edited_relations
        )
        if ok:
            st.success("保存成功。")
            st.session_state.anno = load_annotation(selected_file)

    with st.expander("当前 schema 预览"):
        preview = {
            "image_id": selected_file,
            "paper_id": str(paper_id).strip() if str(paper_id).strip() else os.path.splitext(selected_file)[0],
            "summary": {
                "text": str(summary_text).strip()
            },
            "entities": generated_entities.to_dict(orient="records"),
            "relations": ensure_relation_df(edited_relations).to_dict(orient="records")
        }
        st.code(json.dumps(preview, indent=2, ensure_ascii=False), language="json")


def main():
    st.title("AcademicBench 标注工具")
    st.caption("改进版：支持更大的图片查看区域、缩放查看、复杂图更易标注。")

    img_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
    ])

    if not img_files:
        st.warning("data/images 下没有图片。")
        st.stop()

    with st.sidebar:
        st.header("控制面板")
        selected_file = st.selectbox("选择图片", img_files)

        layout_mode = st.radio(
            "布局模式",
            ["上下布局（推荐）", "左右布局"],
            index=0,
            help="复杂图推荐上下布局：图片更大。"
        )

        show_big_view_first = st.checkbox(
            "先显示大图查看区",
            value=True,
            help="勾选后先看图，再在下面做标注。"
        )

    if "last_loaded_file" not in st.session_state or st.session_state.last_loaded_file != selected_file:
        st.session_state.anno = load_annotation(selected_file)
        st.session_state.last_loaded_file = selected_file

    anno = st.session_state.anno

    if layout_mode == "左右布局":
        col_left, col_right = st.columns([0.55, 0.45])

        if show_big_view_first:
            with col_right:
                render_image_panel(selected_file)
            with col_left:
                render_annotation_panel(anno, selected_file)
        else:
            with col_left:
                render_annotation_panel(anno, selected_file)
            with col_right:
                render_image_panel(selected_file)

    else:
        if show_big_view_first:
            render_image_panel(selected_file)
            st.divider()
            render_annotation_panel(anno, selected_file)
        else:
            render_annotation_panel(anno, selected_file)
            st.divider()
            render_image_panel(selected_file)


if __name__ == "__main__":
    main()
