from __future__ import annotations

from pathlib import Path

import streamlit as st

from timelapse_viewer import (
    CATEGORIES,
    discover_category_records,
    get_available_dates,
    load_display_image,
    pick_record_for_date,
)


@st.cache_data(show_spinner=False)
def get_preview_for_record(record, max_size: int = 1400):
    return load_display_image(record.path, max_size=max_size)


st.set_page_config(page_title="Timelapse Viewer", page_icon="📷", layout="wide")


def render_record_panel(record, label: str | None = None) -> None:
    if record is None:
        st.info("No image found for this selection.")
        return

    st.subheader(label or record.category)
    preview = get_preview_for_record(record, max_size=1400)
    image_bytes = preview.convert("RGB")
    image_path = Path("/tmp") / f"{record.filename}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes.save(image_path)

    st.markdown(
        """
        <style>
        .portrait-frame {
            display: block;
            max-height: 82vh;
            width: auto;
            margin: 0 auto;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<img class="portrait-frame" src="data:image/jpeg;base64,{__import__("base64").b64encode(image_path.read_bytes()).decode("utf-8")}" />',
        unsafe_allow_html=True,
    )
    st.caption(f"{record.filename} • {record.dt.strftime('%Y-%m-%d %H:%M:%S')}")


def build_browse_page(source_dir: str) -> None:
    records_by_category = discover_category_records(source_dir)
    dates = get_available_dates(records_by_category)

    if not dates:
        st.warning("No dated images were found in this folder.")
        return

    st.title("📷 Timelapse browser")
    st.caption(f"Browsing {source_dir} • {len(dates)} dates available")

    date_index = st.slider(
        "Date bar",
        min_value=0,
        max_value=len(dates) - 1,
        value=len(dates) - 1,
        step=1,
        key="browse_date_index",
        label_visibility="visible",
    )
    current_date = dates[date_index]
    st.caption(f"Selected date: {current_date}")

    st.markdown(
        """
        <style>
        div[data-testid="stSlider"] > div[data-baseweb="slider"] {
            min-height: 2.5rem;
        }
        div[data-testid="stSlider"] label {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio("Display mode", ["All categories", "Single category"])
    if mode == "Single category":
        category = st.sidebar.selectbox("Category", list(CATEGORIES))
        categories_to_show = [category]
    else:
        categories_to_show = list(CATEGORIES)

    st.write(f"Selected date: {current_date}")

    columns = st.columns(min(len(categories_to_show), 4))
    for index, category in enumerate(categories_to_show):
        with columns[index]:
            records = records_by_category.get(category, [])
            record = pick_record_for_date(records, current_date)
            render_record_panel(record, label=category)


def build_compare_page(source_dir: str) -> None:
    records_by_category = discover_category_records(source_dir)
    dates = get_available_dates(records_by_category)

    if not dates:
        st.warning("No dated images were found in this folder.")
        return

    st.title("🔄 Compare two dates")
    left_date = st.selectbox("First date", dates, index=0)
    right_date = st.selectbox("Second date", dates, index=len(dates) - 1)

    mode = st.sidebar.radio(
        "Comparison mode", ["All categories", "Single category"], key="compare_mode"
    )
    if mode == "Single category":
        category = st.sidebar.selectbox(
            "Category", list(CATEGORIES), key="compare_category"
        )
        categories_to_show = [category]
    else:
        categories_to_show = list(CATEGORIES)

    if left_date == right_date:
        st.info("Choose two different dates to compare the images side by side.")

    for category in categories_to_show:
        st.subheader(f"{category}")
        left_record = pick_record_for_date(
            records_by_category.get(category, []), left_date
        )
        right_record = pick_record_for_date(
            records_by_category.get(category, []), right_date
        )

        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"### {left_date}")
            render_record_panel(left_record)
        with right_col:
            st.markdown(f"### {right_date}")
            render_record_panel(right_record)


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", ["Browse dates", "Compare dates"])

    root = Path(__file__).resolve().parent
    source_dir = st.sidebar.selectbox("Image source", ["images", "aligned"], index=0)
    source_path = root / source_dir
    if not source_path.exists():
        st.error(f"Selected source directory does not exist: {source_path}")
        st.stop()

    if page == "Browse dates":
        build_browse_page(source_dir)
    else:
        build_compare_page(source_dir)


if __name__ == "__main__":
    main()
