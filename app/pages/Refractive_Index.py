import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils.refractive_index_helper import (
    get_refractive_index,
    get_wavelength_span,
    load_catalog,
    load_material_data,
    search_materials,
)

# 独立运行时的页面配置
try:
    st.set_page_config(page_title="镜片折射率库", page_icon="🔍", layout="wide")
except Exception:
    pass


def _plot_refractive_index(data_list):
    span = get_wavelength_span(data_list)
    if not span:
        return None

    wl_min, wl_max = span
    wl_ratio = wl_max / wl_min if wl_min > 0 else 0
    if wl_ratio > 50:
        wls = np.logspace(np.log10(wl_min), np.log10(wl_max), 400)
    else:
        wls = np.linspace(wl_min, wl_max, 400)

    rows = []
    for wl in wls:
        n, k = get_refractive_index(data_list, wl)
        if n is not None:
            rows.append(
                {
                    "wavelength_nm": wl * 1000,
                    "n": n,
                    "k": k if k is not None else 0.0,
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("wavelength_nm", title="波长 (nm)")
    )
    line_n = base.mark_line(color="#1f77b4").encode(
        y=alt.Y("n", title="折射率 n")
    )
    if (df["k"] > 0).any():
        line_k = base.mark_line(color="#d62728", strokeDash=[6, 4]).encode(
            y=alt.Y("k", title="消光系数 k")
        )
        chart = alt.layer(line_n, line_k).resolve_scale(y="independent")
    else:
        chart = line_n

    return chart.properties(title="折射率与波长曲线").interactive()


def main():
    st.title("🔍 Refractive Index")
    st.caption("数据源：refractiveindex.info")

    root, materials = load_catalog()
    if not root or not materials:
        st.error("未找到折射率数据库，请检查 app/data/refractiveindex.info-database。")
        return

    query = st.text_input("输入关键词（材料/厂家/代号）", placeholder="例如：S-TIH53、BK7、Si")
    _, filtered = search_materials(query)
    if not filtered:
        st.warning("未找到匹配的材料。")
        return

    options = [m["label"] for m in filtered[:200]]
    selected_label = st.selectbox("匹配结果（最多显示 200 条）", options)
    selected_material = next((m for m in filtered if m["label"] == selected_label), filtered[0])

    material_data = load_material_data(selected_material["data_path"])
    if not material_data or "DATA" not in material_data:
        st.error("无法加载材料数据。")
        return

    span = get_wavelength_span(material_data["DATA"])
    default_nm = 976.0
    if span:
        span_nm = (span[0] * 1000, span[1] * 1000)
        st.caption(f"可用波长范围：{span_nm[0]:.0f} - {span_nm[1]:.0f} nm")
        default_nm = max(span_nm[0], min(span_nm[1], default_nm))

    wavelength_nm = st.number_input(
        "波长 (nm)",
        min_value=200.0,
        max_value=20000.0,
        value=float(default_nm),
        step=1.0,
    )
    wavelength_um = wavelength_nm / 1000.0

    n_val, k_val = get_refractive_index(material_data["DATA"], wavelength_um)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("折射率 n", f"{n_val:.6f}" if n_val is not None else "—")
    with col2:
        st.metric("消光系数 k", f"{(k_val or 0):.4e}" if n_val is not None else "—")

    chart = _plot_refractive_index(material_data["DATA"])
    if chart:
        st.altair_chart(chart, use_container_width=True)

    with st.expander("数据来源/备注", expanded=False):
        st.write(f"库路径：{root}")
        st.write(f"材料：{selected_material['label']}")
        if material_data.get("COMMENTS"):
            st.markdown(f"**备注：** {material_data['COMMENTS']}")
        if material_data.get("REFERENCES"):
            st.markdown("**参考文献：**")
            st.caption(material_data["REFERENCES"])


if __name__ == "__main__":
    main()
