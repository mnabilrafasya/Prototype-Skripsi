"""Tema dan kerangka halaman Streamlit."""

from __future__ import annotations

import streamlit as st

from core.konstanta import FOOTER, WARNA_AKSEN

_CSS_DASAR = f"""
.block-container {{
  padding-top: 2rem;
  padding-bottom: 2.5rem;
  max-width: 1220px;
}}
h1 {{
  color: {WARNA_AKSEN};
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-bottom: 0.75rem;
}}
h2 {{
  color: {WARNA_AKSEN};
  font-weight: 600;
  border-bottom: 1px solid #DCE2EA;
  padding-bottom: 0.3rem;
  margin-top: 1.5rem;
  margin-bottom: 0.75rem;
}}
h3, h4 {{
  color: {WARNA_AKSEN};
  font-weight: 600;
}}
p, li {{
  line-height: 1.55;
}}
button[data-baseweb="tab"] {{
  font-weight: 600;
  color: #333333;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: {WARNA_AKSEN};
}}
[data-testid="stMetric"] {{
  background-color: #F8FAFC;
  border: 1px solid #DCE2EA;
  border-radius: 6px;
  padding: 0.7rem 0.9rem;
}}
[data-testid="stMetricLabel"] {{
  color: #334155;
}}
[data-testid="stMetricValue"] {{
  color: {WARNA_AKSEN};
}}
.footer-skripsi {{
  font-size: 0.8rem;
  color: #333333;
  margin-top: 2.25rem;
  border-top: 1px solid #C8CDD3;
  padding-top: 0.6rem;
}}
"""


def terapkan_tema() -> None:
    """Suntikkan CSS tema halaman."""
    st.markdown(f"<style>{_CSS_DASAR}</style>", unsafe_allow_html=True)


def judul(teks: str) -> None:
    """Header halaman: terapkan tema lalu tampilkan hanya judul halaman."""
    terapkan_tema()
    st.title(teks)


def footer() -> None:
    """Footer kecil di bagian bawah halaman."""
    st.markdown(f'<div class="footer-skripsi">{FOOTER}</div>', unsafe_allow_html=True)
