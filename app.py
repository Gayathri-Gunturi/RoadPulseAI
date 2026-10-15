import streamlit as st
import pandas as pd
import cv2
import sqlite3

from PIL import Image
from ultralytics import YOLO
from datetime import datetime


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="RoadPulse",
    page_icon="🛣️",
    layout="wide"
)

# --------------------------------------------------
# Look and feel
#   Concrete-grey page, asphalt sidebar, double yellow
#   centre line as the one decorative motif. Signage-style
#   headings (Barlow Semi Condensed), plain body text
#   (Public Sans). Small 2px corners everywhere, no shadows.
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap');

    :root {
        --concrete: #E6E8E5;
        --paper: #F6F6F3;
        --asphalt: #2A2D30;
        --line: #E8B40F;
        --slate: #5E6469;
        --rule: #C9CDC9;
        --red: #A93226;
        --orange: #B8641A;
        --moss: #44694B;
    }

    html, body, .stApp, .stMarkdown, p, li, label, button, input,
    textarea, [data-baseweb="select"] *, [data-testid="stCaptionContainer"] {
        font-family: 'Public Sans', sans-serif;
    }

    .stApp {
        background-color: var(--concrete);
        color: var(--asphalt);
    }

    /* Keep the sidebar toggle, drop the rest of Streamlit's chrome */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        padding-top: 2.6rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--asphalt);
    }
    [data-testid="stSidebar"] * {
        color: #E6E8E5;
    }
    .brand {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 700;
        font-size: 1.9rem;
        line-height: 1;
        letter-spacing: 0.01em;
        margin-top: 0.4rem;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #A9AEB2 !important;
        margin: 0.35rem 0 1.4rem 0;
    }
    [data-testid="stSidebar"] hr {
        border-color: #44494E;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.1rem;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        padding: 0.45rem 0.6rem;
        border-left: 3px solid transparent;
        border-radius: 0 2px 2px 0;
        margin-left: -0.6rem;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
        border-left-color: var(--line);
        background: #353a3e;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) * {
        font-weight: 600;
        color: #FFFFFF;
    }

    /* ---------- Page header ---------- */
    .page-head h1 {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        font-size: 2.5rem;
        line-height: 1.1;
        color: var(--asphalt);
        margin: 0;
        padding: 0;
    }
    .page-head .centre-line {
        width: 64px;
        height: 7px;
        border-top: 2px solid var(--line);
        border-bottom: 2px solid var(--line);
        margin: 0.7rem 0 0.8rem 0;
    }
    .page-head p {
        color: var(--slate);
        max-width: 60ch;
        margin: 0 0 1.4rem 0;
        font-size: 1rem;
        line-height: 1.5;
    }

    h2, h3 {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        color: var(--asphalt);
    }
    h3 { font-size: 1.35rem; }

    /* ---------- Stats (replace st.metric) ---------- */
    .stat {
        border-left: 3px solid var(--asphalt);
        padding: 0.1rem 0 0.1rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.88rem;
        color: var(--slate);
        margin-bottom: 0.1rem;
    }
    .stat-value {
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        font-size: 2.3rem;
        line-height: 1.15;
        color: var(--asphalt);
    }

    /* ---------- Priority tags ---------- */
    .tag {
        display: inline-block;
        font-family: 'Public Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.12rem 0.55rem;
        border-radius: 2px;
        border: 1px solid currentColor;
        vertical-align: middle;
    }
    .tag-high { color: var(--red); background: #F3DEDB; }
    .tag-med  { color: var(--orange); background: #F5E4D2; }
    .tag-low  { color: var(--moss); background: #DDE8DE; }
    .tag-none { color: var(--slate); background: #E1E3E0; }

    /* ---------- Repair progress ---------- */
    .trail {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 6px;
        margin: 0.6rem 0 1.4rem 0;
    }
    .trail .bar {
        display: block;
        height: 8px;
        background: #CBCFCC;
        border-radius: 1px;
    }
    .trail .name {
        display: block;
        font-size: 0.88rem;
        color: var(--slate);
        margin-top: 0.4rem;
    }
    .trail .done .bar { background: var(--asphalt); }
    .trail .current .bar { background: var(--line); }
    .trail .current .name { color: var(--asphalt); font-weight: 600; }
    .trail .final .bar { background: var(--moss); }
    .trail .final .name { color: var(--moss); font-weight: 600; }

    /* ---------- Empty state ---------- */
    .empty-note {
        border: 1px dashed #9AA0A4;
        border-radius: 2px;
        padding: 1.1rem 1.2rem;
        color: var(--slate);
        background: var(--paper);
        margin-top: 0.6rem;
    }

    /* ---------- Inputs and buttons ---------- */
    div.stButton > button,
    [data-testid="stDownloadButton"] button {
        border-radius: 2px;
        padding: 0.5rem 1.1rem;
        font-weight: 600;
    }
    div.stButton > button {
        background-color: var(--asphalt);
        border: 1px solid var(--asphalt);
        color: #FFFFFF;
    }
    div.stButton > button p { color: #FFFFFF; }
    div.stButton > button:hover {
        background-color: var(--line);
        border-color: var(--line);
    }
    div.stButton > button:hover p { color: var(--asphalt); }

    [data-testid="stDownloadButton"] button {
        background-color: transparent;
        border: 1px solid var(--asphalt);
        color: var(--asphalt);
    }
    [data-testid="stDownloadButton"] button p { color: var(--asphalt); }
    [data-testid="stDownloadButton"] button:hover {
        background-color: var(--asphalt);
    }
    [data-testid="stDownloadButton"] button:hover p { color: #FFFFFF; }

    button:focus-visible {
        outline: 2px solid var(--asphalt);
        outline-offset: 2px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--paper);
        border: 1px dashed #9AA0A4;
        border-radius: 2px;
    }

    [data-baseweb="select"] > div {
        background-color: var(--paper);
        border-radius: 2px;
    }

    [data-testid="stAlert"] {
        border-radius: 2px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--rule);
        border-radius: 2px;
        overflow: hidden;
    }

    hr { border-color: var(--rule); }
</style>
""", unsafe_allow_html=True)


DB_NAME = "roadpulse.db"

REPAIR_STATUSES = [
    "Reported",
    "Under Review",
    "Repair Scheduled",
    "Resolved"
]

PRIORITY_ORDER = [
    "High",
    "Medium",
    "Low",
    "No potholes detected"
]

PRIORITY_COLORS = {
    "High": "#A93226",
    "Medium": "#B8641A",
    "Low": "#44694B",
    "No potholes detected": "#5E6469"
}

STATUS_COLORS = {
    "Reported": "#5E6469",
    "Under Review": "#B8641A",
    "Repair Scheduled": "#2A2D30",
    "Resolved": "#44694B"
}

TABLE_LABELS = {
    "id": st.column_config.NumberColumn("ID", width="small"),
    "inspected_at": st.column_config.TextColumn("Date and time"),
    "image_name": st.column_config.TextColumn("Image"),
    "detection_count": st.column_config.NumberColumn("Potholes found"),
    "highest_confidence": st.column_config.NumberColumn(
        "Top confidence", format="%.2f"
    ),
    "priority": st.column_config.TextColumn("Priority"),
    "repair_status": st.column_config.TextColumn("Repair status"),
}


# ==================================================
# DATABASE
# ==================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspected_at TEXT NOT NULL,
            image_name TEXT,
            detection_count INTEGER,
            highest_confidence REAL,
            priority TEXT
        )
    """)

    # Safely upgrade an existing database.
    # Existing inspection records are preserved.
    columns = [
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(inspections)"
        ).fetchall()
    ]

    if "repair_status" not in columns:
        conn.execute("""
            ALTER TABLE inspections
            ADD COLUMN repair_status TEXT
            NOT NULL DEFAULT 'Reported'
        """)

    conn.commit()
    conn.close()


def save_inspection(image_name, count, confidence, priority):
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO inspections (
            inspected_at,
            image_name,
            detection_count,
            highest_confidence,
            priority
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_name,
        count,
        confidence,
        priority
    ))

    conn.commit()
    conn.close()


def get_inspections():
    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("""
        SELECT
            id,
            inspected_at,
            image_name,
            detection_count,
            highest_confidence,
            priority,
            repair_status
        FROM inspections
        ORDER BY id DESC
    """, conn)

    conn.close()
    return df


def update_repair_status(inspection_id, new_status):
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        UPDATE inspections
        SET repair_status = ?
        WHERE id = ?
    """, (new_status, inspection_id))

    conn.commit()
    conn.close()


# Initialize database before displaying pages.
init_db()


# ==================================================
# LOAD YOLO MODEL
# ==================================================
@st.cache_resource
def load_model():
    return YOLO(
        "https://huggingface.co/peterhdd/"
        "pothole-detection-yolov8/resolve/main/best.pt"
    )


# ==================================================
# UI HELPERS
# ==================================================
def page_header(title, blurb):
    st.markdown(
        f"""
        <div class="page-head">
            <h1>{title}</h1>
            <div class="centre-line"></div>
            <p>{blurb}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def stat(label, value_html):
    return (
        '<div class="stat">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value_html}</div>'
        '</div>'
    )


def priority_tag(priority):
    css = {
        "High": "tag-high",
        "Medium": "tag-med",
        "Low": "tag-low"
    }.get(priority, "tag-none")

    return f'<span class="tag {css}">{priority}</span>'


def status_trail(current_status):
    """Four-step bar showing how far a repair has got."""
    if current_status in REPAIR_STATUSES:
        current_index = REPAIR_STATUSES.index(current_status)
    else:
        current_index = -1

    last_index = len(REPAIR_STATUSES) - 1
    parts = []

    for index, name in enumerate(REPAIR_STATUSES):
        if current_index == last_index:
            css = "final" if index == last_index else "done"
        elif index < current_index:
            css = "done"
        elif index == current_index:
            css = "current"
        else:
            css = ""

        parts.append(
            f'<div class="{css}">'
            f'<span class="bar"></span>'
            f'<span class="name">{name}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="trail">{"".join(parts)}</div>',
        unsafe_allow_html=True
    )


def _color_map(styler, func, subset):
    # pandas 2.1+ renamed applymap to map
    method = getattr(styler, "map", None) or styler.applymap
    return method(func, subset=subset)


def style_table(df):
    """Colour the priority and status text; leave everything else plain."""
    styler = df.style

    if "priority" in df.columns:
        styler = _color_map(
            styler,
            lambda v: (
                f"color: {PRIORITY_COLORS.get(v, '#2A2D30')}; "
                "font-weight: 600"
            ),
            ["priority"]
        )

    if "repair_status" in df.columns:
        styler = _color_map(
            styler,
            lambda v: (
                f"color: {STATUS_COLORS.get(v, '#2A2D30')}; "
                "font-weight: 600"
            ),
            ["repair_status"]
        )

    return styler


def show_table(df):
    config = {
        column: TABLE_LABELS[column]
        for column in df.columns
        if column in TABLE_LABELS
    }

    st.dataframe(
        style_table(df),
        use_container_width=True,
        hide_index=True,
        column_config=config
    )


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.markdown(
    '<div class="brand">RoadPulse</div>'
    '<div class="brand-sub">Pothole inspection log</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Check an image",
        "Past inspections",
        "Summary",
        "Repairs"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()


# ==================================================
# PAGE 1: CHECK AN IMAGE
# ==================================================
if page == "Check an image":

    page_header(
        "Check an image",
        "Upload a photo of the road and the model will mark any potholes "
        "it finds. Every result is saved to the inspection history."
    )

    confidence_threshold = st.sidebar.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values catch more potholes but also more false alarms."
    )

    uploaded_file = st.file_uploader(
        "Road photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Your photo")
            st.image(image, use_container_width=True)

        if st.button("Find potholes", type="primary"):

            with st.spinner("Looking at the image..."):

                try:
                    model = load_model()

                    results = model.predict(
                        source=image,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    result = results[0]
                    boxes = result.boxes
                    count = len(boxes)

                    annotated_bgr = result.plot()
                    annotated_rgb = cv2.cvtColor(
                        annotated_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    image_area = image.width * image.height
                    area_ratios = []
                    detection_rows = []

                    for index, box in enumerate(boxes, start=1):
                        confidence = float(box.conf[0])

                        x1, y1, x2, y2 = [
                            float(value)
                            for value in box.xyxy[0]
                        ]

                        box_area = (
                            max(0, x2 - x1)
                            * max(0, y2 - y1)
                        )

                        area_ratio = box_area / image_area
                        area_ratios.append(area_ratio)

                        detection_rows.append({
                            "Pothole": index,
                            "Confidence": round(confidence, 3),
                            "Share of image (%)": round(
                                area_ratio * 100, 2
                            )
                        })

                    # Simple priority rules, kept visible so they can be checked
                    largest_area = max(area_ratios, default=0)

                    if count == 0:
                        priority = "No potholes detected"
                        explanation = (
                            "Nothing was found above the minimum "
                            "confidence. That doesn't prove the road is "
                            "undamaged, so try lowering the threshold or "
                            "using a closer photo."
                        )

                    elif count >= 3 or largest_area >= 0.10:
                        priority = "High"
                        explanation = (
                            "Three or more potholes, or one covering a "
                            "large part of the photo. Worth reviewing first."
                        )

                    elif count >= 2 or largest_area >= 0.04:
                        priority = "Medium"
                        explanation = (
                            "Two potholes, or one of moderate size. "
                            "A closer look is a good idea."
                        )

                    else:
                        priority = "Low"
                        explanation = (
                            "One small pothole. Confirm it in the photo "
                            "before deciding on a repair."
                        )

                    highest_confidence = max(
                        (
                            row["Confidence"]
                            for row in detection_rows
                        ),
                        default=0
                    )

                    # Save each inspection. New records default to Reported.
                    save_inspection(
                        image_name=uploaded_file.name,
                        count=count,
                        confidence=highest_confidence,
                        priority=priority
                    )

                    with right:
                        st.subheader("What the model found")
                        st.image(
                            annotated_rgb,
                            caption="Boxes show detected potholes",
                            use_container_width=True
                        )

                    st.success(
                        "Saved to the inspection history with status "
                        "Reported."
                    )

                    st.divider()

                    col1, col2, col3 = st.columns(3)

                    col1.markdown(
                        stat("Potholes found", count),
                        unsafe_allow_html=True
                    )
                    col2.markdown(
                        stat(
                            "Top confidence",
                            f"{highest_confidence:.0%}"
                        ),
                        unsafe_allow_html=True
                    )
                    col3.markdown(
                        stat("Priority", priority_tag(priority)),
                        unsafe_allow_html=True
                    )

                    st.write(explanation)

                    if detection_rows:
                        st.subheader("Each detection")

                        detection_df = pd.DataFrame(detection_rows)

                        st.dataframe(
                            detection_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv = detection_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download as CSV",
                            data=csv,
                            file_name="roadpulse_detection_report.csv",
                            mime="text/csv"
                        )

                except Exception as error:
                    st.error(
                        f"Couldn't analyze this image: {error}. "
                        "Check that it's a valid JPG or PNG and try again."
                    )

    else:
        st.markdown(
            '<div class="empty-note">No photo yet. Upload a JPG or PNG '
            'of a road surface to start.</div>',
            unsafe_allow_html=True
        )


# ==================================================
# PAGE 2: PAST INSPECTIONS
# ==================================================
elif page == "Past inspections":

    page_header(
        "Past inspections",
        "Every image checked so far, newest first."
    )

    history_df = get_inspections()

    if history_df.empty:
        st.markdown(
            '<div class="empty-note">Nothing here yet. Check an image '
            'and it will show up in this list.</div>',
            unsafe_allow_html=True
        )
    else:
        show_table(history_df)

        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download history as CSV",
            data=history_csv,
            file_name="roadpulse_inspection_history.csv",
            mime="text/csv"
        )


# ==================================================
# PAGE 3: SUMMARY
# ==================================================
elif page == "Summary":

    page_header(
        "Summary",
        "Totals across all saved inspections."
    )

    history_df = get_inspections()

    if history_df.empty:
        st.markdown(
            '<div class="empty-note">No data to summarize yet. Check a '
            'few images first.</div>',
            unsafe_allow_html=True
        )
    else:
        total_inspections = len(history_df)
        total_detections = int(
            history_df["detection_count"].sum()
        )
        high_priority = int(
            (history_df["priority"] == "High").sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.markdown(
            stat("Images checked", total_inspections),
            unsafe_allow_html=True
        )
        col2.markdown(
            stat("Potholes found", total_detections),
            unsafe_allow_html=True
        )
        col3.markdown(
            stat("High priority", high_priority),
            unsafe_allow_html=True
        )

        st.divider()

        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            st.subheader("By repair status")

            status_counts = (
                history_df["repair_status"]
                .value_counts()
                .reindex(REPAIR_STATUSES, fill_value=0)
                .rename_axis("Repair status")
                .reset_index(name="Inspections")
            )

            st.bar_chart(
                status_counts.set_index("Repair status"),
                color="#2A2D30"
            )

        with chart_right:
            st.subheader("By priority")

            priority_counts = (
                history_df["priority"]
                .value_counts()
                .reindex(PRIORITY_ORDER, fill_value=0)
                .rename_axis("Priority")
                .reset_index(name="Inspections")
            )

            st.bar_chart(
                priority_counts.set_index("Priority"),
                color="#E8B40F"
            )

        st.caption(
            "Priority comes from simple rules on pothole count and size, "
            "so treat it as a rough guide."
        )


# ==================================================
# PAGE 4: REPAIRS
# ==================================================
elif page == "Repairs":

    page_header(
        "Repairs",
        "Pick an inspection and move it along as work gets done."
    )

    history_df = get_inspections()

    if history_df.empty:
        st.markdown(
            '<div class="empty-note">No inspections to manage yet. '
            'Check an image to create one.</div>',
            unsafe_allow_html=True
        )

    else:
        st.subheader("Update an inspection")

        inspection_ids = history_df["id"].astype(int).tolist()

        pick_col, info_col = st.columns([1, 2], gap="large")

        with pick_col:
            selected_id = st.selectbox(
                "Inspection ID",
                inspection_ids
            )

        selected_record = history_df[
            history_df["id"] == selected_id
        ].iloc[0]

        current_status = selected_record["repair_status"]

        with info_col:
            st.write(f"**Image:** {selected_record['image_name']}")
            st.markdown(
                f"**Priority:** {priority_tag(selected_record['priority'])}",
                unsafe_allow_html=True
            )

        status_trail(current_status)

        # Support old or unexpected status values safely.
        default_index = (
            REPAIR_STATUSES.index(current_status)
            if current_status in REPAIR_STATUSES
            else 0
        )

        status_col, button_col = st.columns([1, 2], gap="large")

        with status_col:
            new_status = st.selectbox(
                "New status",
                REPAIR_STATUSES,
                index=default_index,
                key=f"new_status_{selected_id}"
            )

        with button_col:
            st.markdown("<div style='height: 1.75rem'></div>",
                        unsafe_allow_html=True)

            if st.button("Save status", type="primary"):

                if new_status == current_status:
                    st.info("This inspection already has that status.")
                else:
                    update_repair_status(
                        selected_id,
                        new_status
                    )

                    st.success(
                        f"Inspection #{selected_id} is now "
                        f"'{new_status}'."
                    )

                    st.rerun()

        st.divider()
        st.subheader("All inspections")

        show_table(
            history_df[
                [
                    "id",
                    "inspected_at",
                    "image_name",
                    "detection_count",
                    "priority",
                    "repair_status"
                ]
            ]
        )


# ==================================================
# SIDEBAR NOTE (added last so it sits under any controls)
# ==================================================
st.sidebar.caption(
    "Results are model estimates, not a road-safety assessment. "
    "Check each one by eye before acting on it."
)