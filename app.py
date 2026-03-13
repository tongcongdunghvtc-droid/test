import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import io
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CAU HINH TRANG
# ============================================================
st.set_page_config(
    page_title="SEO Content Mapping Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    "<style>"
    ".main .block-container {"
    "  padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;"
    "}"
    ".main-header {"
    "  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
    "  padding: 2rem; border-radius: 16px; color: white;"
    "  text-align: center; margin-bottom: 2rem;"
    "  box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);"
    "}"
    ".main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }"
    ".main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }"
    "#MainMenu {visibility: hidden;}"
    "footer {visibility: hidden;}"
    "header {visibility: hidden;}"
    "</style>",
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================
if "tool1_result" not in st.session_state:
    st.session_state.tool1_result = None
if "tool1_trash" not in st.session_state:
    st.session_state.tool1_trash = None
if "tool2_result" not in st.session_state:
    st.session_state.tool2_result = None

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-header">'
    '<h1>🔍 SEO Content Mapping Tool</h1>'
    '<p>Lọc từ khóa thông minh &amp; Gom nhóm bài viết bằng AI</p>'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
DEFAULT_API_KEY = "AIzaSyBSPo-XImF7uXzZxpRTclt6-hSRxuS-U5g"

with st.sidebar:
    st.markdown("## ⚙️ Điều khiển")
    st.markdown("---")
    tool_choice = st.radio(
        "🎯 Chọn công cụ",
        options=[
            "Công cụ 1: Lọc & Gom từ khóa",
            "Công cụ 2: Gom bài viết bằng AI",
            "Pipeline: Chạy cả 2"
        ],
        index=0,
    )
    st.markdown("---")
    with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
        st.markdown(
            "**Công cụ 1** - Lọc & Gom từ khóa:\n"
            "- Upload file CSV từ Ahrefs\n"
            "- Cấu hình hạt giống mục tiêu & nhiễu\n"
            "- AI sẽ lọc rác và gom nhóm ngữ nghĩa\n\n"
            "**Công cụ 2** - Gom bài viết bằng Gemini:\n"
            "- Upload file Excel hoặc dùng output từ Công cụ 1\n"
            "- Cần có cột Từ khóa và Volume\n\n"
            "**Pipeline** - Chạy liên tục:\n"
            "- Upload CSV, chạy Công cụ 1 xong tự động sang Công cụ 2"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_keyword_column(df):
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["keyword", "keywords", "từ khóa", "từ khoá", "tu khoa"]:
            return c
    for c in df.columns:
        cl = c.lower().strip()
        if "keyword" in cl or "từ khóa" in cl or "từ khoá" in cl:
            return c
    return None


def find_volume_column(df):
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["volume", "search volume", "vol"]:
            return c
    for c in df.columns:
        cl = c.lower().strip()
        if "volume" in cl:
            return c
    return None


def safe_to_numeric_series(df, col_name):
    series = df[col_name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce").fillna(0)


def to_excel_bytes(df_main, df_trash=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Content Map", index=False)
        if df_trash is not None and len(df_trash) > 0:
            df_trash.to_excel(
                writer, sheet_name="Từ Khóa Bị Loại", index=False
            )
    return output.getvalue()


def to_excel_bytes_single(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Content Map AI", index=False)
    return output.getvalue()


# ============================================================
# CONG CU 1: LOC & GOM TU KHOA
# ============================================================
def run_tool1(uploaded_file, target_seeds, noise_seeds,
              distance_threshold, margin):
    from sentence_transformers import SentenceTransformer, util
    from sklearn.cluster import AgglomerativeClustering

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.markdown(
        "📂 **Bước 1/5:** Đang đọc dữ liệu đầu vào..."
    )
    progress_bar.progress(5)

    try:
        try:
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig",
                on_bad_lines="skip"
            )
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding="utf-16",
                sep="\t",
                engine="python",
                on_bad_lines="skip"
            )
    except Exception as e:
        st.error("❌ Không thể đọc file: " + str(e))
        return None, None

    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")

    with st.expander("🔍 Debug: Cấu trúc file CSV", expanded=False):
        st.write("Các cột tìm thấy:", list(df.columns))
        st.dataframe(df.head())

    col_kw = find_keyword_column(df)
    col_vol = find_volume_column(df)

    if not col_kw or not col_vol:
        st.error(
            "❌ Không tìm thấy cột Keyword hoặc Volume. "
            "Các cột: " + str(list(df.columns))
        )
        return None, None

    st.success(
        "✅ Nhận diện: Keyword = **" + col_kw
        + "**, Volume = **" + col_vol + "**"
    )

    df = df.rename(columns={col_kw: "Keyword", col_vol: "Volume"})
    df["Volume"] = safe_to_numeric_series(df, "Volume")
    df = df.dropna(subset=["Keyword"])
    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    df = df[df["Keyword"] != ""]
    keywords = df["Keyword"].tolist()

    st.info(
        "📊 Tổng từ khóa đầu vào: **" + str(len(keywords)) + "**"
    )
    progress_bar.progress(15)

    if len(keywords) == 0:
        st.error("❌ File không có từ khóa hợp lệ!")
        return None, None

    status_text.markdown(
        "🧠 **Bước 2/5:** Đang tải AI (MPNet)..."
    )
    progress_bar.progress(20)
    smodel = SentenceTransformer(
        "paraphrase-multilingual-mpnet-base-v2"
    )
    progress_bar.progress(40)

    status_text.markdown(
        "🔬 **Bước 3/5:** AI đang phân định ranh giới ngữ nghĩa..."
    )
    kw_vecs = smodel.encode(
        keywords,
        batch_size=64,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    target_vecs = smodel.encode(target_seeds, convert_to_tensor=True)
    noise_vecs = smodel.encode(noise_seeds, convert_to_tensor=True)

    target_scores = (
        util.cos_sim(kw_vecs, target_vecs)
        .max(dim=1).values.tolist()
    )
    noise_scores = (
        util.cos_sim(kw_vecs, noise_vecs)
        .max(dim=1).values.tolist()
    )

    clean_indices = []
    trash_indices = []
    for i in range(len(keywords)):
        if target_scores[i] > (noise_scores[i] + margin):
            clean_indices.append(i)
        else:
            trash_indices.append(i)

    df_clean = df.iloc[clean_indices].copy().reset_index(drop=True)
    df_trash = df.iloc[trash_indices].copy()
    clean_kw_vecs = kw_vecs[clean_indices]
    progress_bar.progress(60)

    c1, c2 = st.columns(2)
    with c1:
        st.success(
            "✅ Giữ lại: **" + str(len(df_clean)) + "** từ khóa"
        )
    with c2:
        st.warning(
            "🗑️ Loại bỏ: **" + str(len(df_trash)) + "** từ khóa"
        )

    if len(df_clean) == 0:
        st.error("❌ Không còn từ khóa nào sau khi lọc!")
        return None, None

    status_text.markdown(
        "🧩 **Bước 4/5:** Gom nhóm ngữ nghĩa..."
    )
    cluster_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    df_clean["Cluster_ID"] = cluster_model.fit_predict(
        clean_kw_vecs.cpu().numpy()
    )
    progress_bar.progress(80)

    status_text.markdown("📊 **Bước 5/5:** Đang tạo báo cáo...")
    content_map_data = []
    cluster_volumes = (
        df_clean.groupby("Cluster_ID")["Volume"]
        .sum().reset_index()
    )
    cluster_volumes = cluster_volumes.sort_values(
        by="Volume", ascending=False
    )

    for cid in cluster_volumes["Cluster_ID"]:
        group = df_clean[
            df_clean["Cluster_ID"] == cid
        ].sort_values(by="Volume", ascending=False)
        focus_keyword = group.iloc[0]["Keyword"]
        total_volume = group["Volume"].sum()

        for idx in range(len(group)):
            row = group.iloc[idx]
            is_main = (idx == 0)
            content_map_data.append({
                "Chủ Đề (Tên Bài)": focus_keyword,
                "Phân Loại": (
                    "1 - Keyword Chính" if is_main
                    else "2 - Keyword Phụ"
                ),
                "Từ Khóa": row["Keyword"],
                "Volume": row["Volume"],
                "Tổng Traffic Nhóm": (
                    total_volume if is_main else None
                )
            })

    df_final = pd.DataFrame(content_map_data)
    progress_bar.progress(100)
    status_text.markdown("✅ **Hoàn thành!**")

    df_trash_out = df_trash[["Keyword", "Volume"]].sort_values(
        by="Volume", ascending=False
    )
    return df_final, df_trash_out


# ============================================================
# GIAO DIEN CONG CU 1
# ============================================================
if "Công cụ 1" in tool_choice:
    st.markdown("## 🧹 Công cụ 1: Lọc & Gom từ khóa Semantic")
    st.markdown(
        "Upload file CSV từ Ahrefs, AI sẽ lọc rác "
        "và gom nhóm từ khóa."
    )

    col_upload, col_config = st.columns([1, 1])

    with col_upload:
        st.markdown("### 📁 Upload File")
        uploaded_file = st.file_uploader(
            "Chọn file CSV từ Ahrefs",
            type=["csv"],
            key="tool1_upload"
        )

    with col_config:
        st.markdown("### 🎯 Cấu hình hạt giống")
        target_input = st.text_area(
            "🎯 Hạt giống MỤC TIÊU (mỗi dòng 1 từ)",
            value="kế toán\nhóa đơn",
            height=100,
        )
        noise_default = (
            "học sinh\nsinh viên\ngiải trí\n"
            "mạng xã hội\nfacebook\ngmail\n"
            "game\nphim ảnh\nhoa\nhack\n"
            "lừa đảo\ntải nhạc\nmua sắm\n"
            "đời sống\nnông nghiệp"
        )
        noise_input = st.text_area(
            "🚫 Hạt giống NHIỄU (mỗi dòng 1 từ)",
            value=noise_default,
            height=150,
        )

    with st.expander("⚙️ Tham số nâng cao", expanded=False):
        adv1, adv2 = st.columns(2)
        with adv1:
            distance_threshold = st.slider(
                "Ngưỡng khoảng cách gom nhóm",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05
            )
        with adv2:
            margin = st.slider(
                "Biên độ an toàn (Margin)",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.01
            )

    st.markdown("---")

    if uploaded_file is not None:
        if st.button(
            "🚀 Bắt đầu lọc & gom từ khóa",
            type="primary",
            use_container_width=True
        ):
            target_seeds = [
                s.strip()
                for s in target_input.strip().split("\n")
                if s.strip()
            ]
            noise_seeds = [
                s.strip()
                for s in noise_input.strip().split("\n")
                if s.strip()
            ]
            if not target_seeds:
                st.error("❌ Nhập ít nhất 1 hạt giống mục tiêu!")
            else:
                with st.spinner("Đang xử lý..."):
                    df_result, df_trash = run_tool1(
                        uploaded_file,
                        target_seeds,
                        noise_seeds,
                        distance_threshold,
                        margin
                    )
                if df_result is not None:
                    st.session_state.tool1_result = df_result
                    st.session_state.tool1_trash = df_trash

    # Hien thi ket qua Cong cu 1
    if st.session_state.tool1_result is not None:
        st.markdown("### 📊 Kết quả")
        df_res = st.session_state.tool1_result
        df_tr = st.session_state.tool1_trash

        num_clusters = df_res["Chủ Đề (Tên Bài)"].nunique()
        trash_count = len(df_tr) if df_tr is not None else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("📝 Số nhóm bài", str(num_clusters))
        m2.metric("✅ Từ khóa giữ", str(len(df_res)))
        m3.metric("🗑️ Từ khóa loại", str(trash_count))

        tab1, tab2 = st.tabs(
            ["📗 Content Map", "🗑️ Từ khóa bị loại"]
        )
        with tab1:
            st.dataframe(
                df_res, use_container_width=True, height=400
            )
        with tab2:
            if df_tr is not None and len(df_tr) > 0:
                st.dataframe(
                    df_tr, use_container_width=True, height=400
                )
            else:
                st.info("Không có từ khóa nào bị loại.")

        excel_bytes = to_excel_bytes(df_res, df_tr)
        st.download_button(
            label="📥 Tải xuống file Excel",
            data=excel_bytes,
            file_name="Content_Map_Semantic.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True
        )

        st.markdown("---")
        st.info(
            "💡 **Mẹo:** Chuyển sang **Công cụ 2** hoặc "
            "**Pipeline** ở sidebar để dùng kết quả này "
            "làm đầu vào cho Gemini AI."
        )

# === KET THUC PHAN 1/2 ===
# Phan tiep theo gom: Cong cu 2, Pipeline, render_api_config,
# render_tool2_result
# ============================================================
# CONG CU 2: GOM BAI VIET BANG GEMINI
# ============================================================
def run_tool2(df_input, api_key, batch_size):
    import google.generativeai as genai

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.markdown(
        "🔑 **Bước 1/3:** Đang kết nối Gemini AI..."
    )
    genai.configure(api_key=api_key)

    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        model_name = ""
        for name in [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash"
        ]:
            if name in available_models:
                model_name = name
                break
        if not model_name and available_models:
            model_name = available_models[0]
        if not model_name:
            st.error("❌ API Key không hợp lệ.")
            return None

        st.info("🤖 Model: **" + model_name + "**")
        ai_model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.1}
        )
    except Exception as e:
        st.error("❌ Lỗi kết nối API: " + str(e))
        return None

    progress_bar.progress(10)

    status_text.markdown(
        "📂 **Bước 2/3:** Đang chuẩn bị dữ liệu..."
    )
    df_input.columns = df_input.columns.astype(str).str.strip()

    col_kw = find_keyword_column(df_input)
    col_vol = find_volume_column(df_input)

    if not col_kw or not col_vol:
        st.error(
            "❌ Không tìm thấy cột Từ khóa và Volume. "
            "Các cột: " + str(list(df_input.columns))
        )
        return None

    df_work = df_input.rename(
        columns={col_kw: "Từ Khóa", col_vol: "Volume"}
    )
    df_work["Volume"] = safe_to_numeric_series(df_work, "Volume")
    keywords_data = df_work[["Từ Khóa", "Volume"]].to_dict("records")
    total_kw = len(keywords_data)

    st.info("📊 Tổng từ khóa: **" + str(total_kw) + "**")
    progress_bar.progress(15)

    status_text.markdown(
        "🤖 **Bước 3/3:** Đang gửi cho Gemini phân tích..."
    )
    all_articles = []
    total_batches = (
        (total_kw // batch_size)
        + (1 if total_kw % batch_size else 0)
    )
    batch_progress = st.empty()

    for i in range(0, total_kw, batch_size):
        batch = keywords_data[i: i + batch_size]
        batch_num = i // batch_size + 1

        batch_progress.markdown(
            "⏳ Batch **" + str(batch_num)
            + "/" + str(total_batches) + "**..."
        )
        current_progress = 15 + int(
            (batch_num / total_batches) * 80
        )
        progress_bar.progress(min(current_progress, 95))

        lines = []
        for item in batch:
            lines.append(
                "- " + str(item["Từ Khóa"])
                + " (Volume: "
                + str(int(item["Volume"])) + ")"
            )
        input_data_str = "\n".join(lines)

        prompt = (
            "Bạn là Trưởng phòng SEO kỹ thuật cực kỳ khắt khe. "
            "Nhiệm vụ: gom nhóm từ khóa thành Bài viết.\n\n"
            "LUẬT GOM NHÓM:\n"
            "1. CÙNG URL: 2 từ khóa chỉ chung nhóm nếu "
            "người dùng muốn đọc ĐÚNG CÙNG 1 BÀI VIẾT.\n"
            "2. CHỐNG GOM CƯỠNG ÉP: Ý nghĩa khác = tách.\n"
            "3. QUYỀN CÔ LẬP: Không liên quan = đứng 1 mình.\n"
            "4. GIỮ NGUYÊN: 100% từ khóa và volume.\n\n"
            "DANH SÁCH TỪ KHÓA:\n"
            + input_data_str + "\n\n"
            "OUTPUT: Chỉ xuất 1 mảng JSON, không code block.\n"
            '[\n'
            '  {\n'
            '    "intent": "Định nghĩa / Hướng dẫn / '
            'Thương mại / Phần mềm",\n'
            '    "main_keyword": "từ khóa chính",\n'
            '    "main_volume": 1000,\n'
            '    "sub_keywords": [\n'
            '      {"keyword": "từ khóa phụ", "volume": 500}\n'
            '    ]\n'
            '  }\n'
            ']'
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ai_model.generate_content(prompt)
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                batch_articles = json.loads(response_text)
                all_articles.extend(batch_articles)
                time.sleep(5)
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    batch_progress.warning(
                        "⚠️ Rate limit Batch "
                        + str(batch_num) + " (Lần "
                        + str(attempt + 1) + "). Chờ 40s..."
                    )
                    time.sleep(40)
                else:
                    batch_progress.error(
                        "❌ Lỗi Batch " + str(batch_num)
                        + ": " + str(e)
                    )
                    break

    if not all_articles:
        st.error("❌ AI không trả về dữ liệu. Thử lại!")
        return None

    status_text.markdown("📊 Đang xuất báo cáo...")
    content_map_data = []

    for article in all_articles:
        intent = article.get("intent", "Chung")
        main_kw = article.get("main_keyword", "")
        main_vol = article.get("main_volume", 0)
        sub_kws = article.get("sub_keywords", [])
        sub_vol_sum = sum(
            [s.get("volume", 0) for s in sub_kws]
        )
        total_volume = main_vol + sub_vol_sum

        content_map_data.append({
            "Search Intent": intent,
            "Từ Khóa Chính (H1)": main_kw,
            "Phân Loại": "1 - Keyword Chính",
            "Từ Khóa": main_kw,
            "Volume": main_vol,
            "Tổng Volume Bài": total_volume
        })
        for sub in sub_kws:
            content_map_data.append({
                "Search Intent": intent,
                "Từ Khóa Chính (H1)": main_kw,
                "Phân Loại": "2 - Keyword Phụ",
                "Từ Khóa": sub.get("keyword", ""),
                "Volume": sub.get("volume", 0),
                "Tổng Volume Bài": None
            })

    df_output = pd.DataFrame(content_map_data)
    df_output["_sort"] = df_output.groupby(
        "Từ Khóa Chính (H1)"
    )["Volume"].transform("sum")
    df_output = df_output.sort_values(
        by=["_sort", "Từ Khóa Chính (H1)", "Phân Loại"],
        ascending=[False, True, True]
    )
    df_output = df_output.drop(columns=["_sort"])

    progress_bar.progress(100)
    status_text.markdown("✅ **Hoàn thành!**")
    return df_output


# ============================================================
# WIDGET: CAU HINH API
# ============================================================
def render_api_config():
    st.markdown("### 🔑 Cấu hình API Gemini")
    api_option = st.radio(
        "Chọn API Key",
        options=[
            "Sử dụng API mặc định",
            "Nhập API Key của tôi"
        ],
        index=0,
        horizontal=True,
    )
    if "Nhập API" in api_option:
        api_key = st.text_input(
            "Nhập Google AI Studio API Key",
            type="password",
            placeholder="AIzaSy...",
        )
        if not api_key:
            st.warning("⚠️ Vui lòng nhập API Key.")
            return None, None
    else:
        api_key = DEFAULT_API_KEY
        st.success("✅ Đang dùng API Key mặc định.")

    batch_size = st.slider(
        "Số từ khóa mỗi batch",
        min_value=20,
        max_value=150,
        value=80,
        step=10,
        help="Khuyến nghị: 60-100"
    )
    return api_key, batch_size


# ============================================================
# WIDGET: HIEN THI KET QUA CONG CU 2
# ============================================================
def render_tool2_result(df_result):
    st.markdown("### 📊 Kết quả gom bài viết bằng AI")
    num_articles = df_result["Từ Khóa Chính (H1)"].nunique()
    total_kw = len(df_result)
    total_vol = df_result["Volume"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("📝 Số bài viết", str(num_articles))
    m2.metric("🔑 Tổng từ khóa", str(total_kw))
    m3.metric("📈 Tổng Volume", str(int(total_vol)))

    st.dataframe(
        df_result, use_container_width=True, height=500
    )

    excel_bytes = to_excel_bytes_single(df_result)
    st.download_button(
        label="📥 Tải xuống Content Map AI (Excel)",
        data=excel_bytes,
        file_name="Content_Map_Gemini.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True
    )


# ============================================================
# GIAO DIEN CONG CU 2
# ============================================================
if "Công cụ 2" in tool_choice:
    st.markdown("## 🤖 Công cụ 2: Gom bài viết bằng Gemini AI")
    st.markdown(
        "Upload file Excel chứa từ khóa đã lọc, "
        "Gemini AI sẽ gom thành bài viết chi tiết."
    )

    api_key, batch_size = render_api_config()

    st.markdown("---")
    st.markdown("### 📁 Nguồn dữ liệu đầu vào")

    data_source = st.radio(
        "Chọn nguồn dữ liệu",
        options=[
            "📤 Upload file Excel mới",
            "♻️ Dùng kết quả từ Công cụ 1"
        ],
        index=0,
        horizontal=True,
    )

    df_for_tool2 = None

    if "Upload" in data_source:
        uploaded_excel = st.file_uploader(
            "Chọn file Excel (cần có cột Từ khóa và Volume)",
            type=["xlsx", "xls"],
            key="tool2_upload"
        )
        if uploaded_excel is not None:
            try:
                try:
                    df_for_tool2 = pd.read_excel(
                        uploaded_excel,
                        sheet_name="Content Map"
                    )
                except Exception:
                    df_for_tool2 = pd.read_excel(uploaded_excel)
                st.success(
                    "✅ Đã đọc file: **"
                    + str(len(df_for_tool2))
                    + "** dòng"
                )
                with st.expander(
                    "👀 Xem trước dữ liệu", expanded=False
                ):
                    st.dataframe(df_for_tool2.head(10))
            except Exception as e:
                st.error("❌ Không thể đọc file: " + str(e))
    else:
        if st.session_state.tool1_result is not None:
            df_for_tool2 = st.session_state.tool1_result.copy()
            st.success(
                "✅ Đã lấy kết quả từ Công cụ 1: **"
                + str(len(df_for_tool2)) + "** dòng"
            )
            with st.expander(
                "👀 Xem trước dữ liệu", expanded=False
            ):
                st.dataframe(df_for_tool2.head(10))
        else:
            st.warning(
                "⚠️ Chưa có kết quả từ Công cụ 1. "
                "Hãy chạy Công cụ 1 trước hoặc upload file."
            )

    st.markdown("---")

    if df_for_tool2 is not None and api_key is not None:
        if st.button(
            "🤖 Bắt đầu gom bài viết bằng AI",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Đang xử lý với Gemini AI..."):
                df_tool2_result = run_tool2(
                    df_for_tool2, api_key, batch_size
                )
            if df_tool2_result is not None:
                st.session_state.tool2_result = df_tool2_result

    if st.session_state.tool2_result is not None:
        render_tool2_result(st.session_state.tool2_result)


# ============================================================
# GIAO DIEN PIPELINE
# ============================================================
if "Pipeline" in tool_choice:
    st.markdown("## 🔄 Pipeline: Lọc từ khóa + Gom bài viết")
    st.markdown(
        "Upload file CSV, chạy Công cụ 1 (lọc & gom) "
        "rồi tự động chuyển sang Công cụ 2 (Gemini AI)."
    )

    st.markdown("### 📁 Bước 1: Upload & Cấu hình Công cụ 1")

    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        p_uploaded = st.file_uploader(
            "Chọn file CSV từ Ahrefs",
            type=["csv"],
            key="pipeline_upload"
        )

    with p_col2:
        p_target = st.text_area(
            "🎯 Hạt giống MỤC TIÊU",
            value="kế toán\nhóa đơn",
            height=80,
            key="p_target"
        )
        p_noise_default = (
            "học sinh\nsinh viên\ngiải trí\n"
            "mạng xã hội\nfacebook\ngmail\n"
            "game\nphim ảnh\nhoa\nhack\n"
            "lừa đảo\ntải nhạc\nmua sắm\n"
            "đời sống\nnông nghiệp"
        )
        p_noise = st.text_area(
            "🚫 Hạt giống NHIỄU",
            value=p_noise_default,
            height=120,
            key="p_noise"
        )

    with st.expander("⚙️ Tham số nâng cao", expanded=False):
        pp1, pp2 = st.columns(2)
        with pp1:
            p_dist = st.slider(
                "Ngưỡng gom nhóm",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05,
                key="p_dist"
            )
        with pp2:
            p_margin = st.slider(
                "Biên độ an toàn",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.01,
                key="p_margin"
            )

    st.markdown("---")
    st.markdown("### 🔑 Bước 2: Cấu hình Gemini API")
    p_api_key, p_batch_size = render_api_config()

    st.markdown("---")

    if p_uploaded is not None and p_api_key is not None:
        if st.button(
            "🚀 Chạy Pipeline (Công cụ 1 → Công cụ 2)",
            type="primary",
            use_container_width=True
        ):
            p_target_seeds = [
                s.strip()
                for s in p_target.strip().split("\n")
                if s.strip()
            ]
            p_noise_seeds = [
                s.strip()
                for s in p_noise.strip().split("\n")
                if s.strip()
            ]

            if not p_target_seeds:
                st.error("❌ Nhập ít nhất 1 hạt giống mục tiêu!")
            else:
                st.markdown("#### 🧹 Đang chạy Công cụ 1...")
                with st.spinner("Công cụ 1 đang xử lý..."):
                    df_result, df_trash = run_tool1(
                        p_uploaded,
                        p_target_seeds,
                        p_noise_seeds,
                        p_dist,
                        p_margin
                    )

                if df_result is not None:
                    st.session_state.tool1_result = df_result
                    st.session_state.tool1_trash = df_trash

                    excel_bytes_1 = to_excel_bytes(
                        df_result, df_trash
                    )
                    st.download_button(
                        label=(
                            "📥 Tải kết quả Công cụ 1 (Excel)"
                        ),
                        data=excel_bytes_1,
                        file_name="Pipeline_Step1_Semantic.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.markdown(
                        "#### 🤖 Đang chạy Công cụ 2 "
                        "(Gemini AI)..."
                    )
                    with st.spinner(
                        "Công cụ 2 đang xử lý với Gemini..."
                    ):
                        df_tool2_result = run_tool2(
                            df_result,
                            p_api_key,
                            p_batch_size
                        )

                    if df_tool2_result is not None:
                        st.session_state.tool2_result = (
                            df_tool2_result
                        )
                        render_tool2_result(df_tool2_result)

    elif p_uploaded is None:
        st.info("📁 Vui lòng upload file CSV để bắt đầu.")

# === KET THUC FILE app.py ===
