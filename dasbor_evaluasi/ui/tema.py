"""Tema, kerangka halaman, dan Mode Presentasi."""

from __future__ import annotations

import streamlit as st

from core.konstanta import FOOTER, WARNA_AKSEN

KUNCI_PRESENTASI = "mode_presentasi"

_CSS_DASAR = f"""
h1, h2, h3 {{ color: {WARNA_AKSEN}; }}
.footer-skripsi {{ font-size: 0.75rem; color: #333333; margin-top: 2rem;
  border-top: 1px solid #C8CDD3; padding-top: 0.5rem; }}
"""

# Huruf diperbesar (20 px = 15 pt) dan sidebar disembunyikan.
_CSS_PRESENTASI = """
html { font-size: 20px; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none; }
.footer-skripsi { font-size: 1rem; }
"""


def presentasi_aktif() -> bool:
    """True bila Mode Presentasi menyala."""
    return bool(st.session_state.get(KUNCI_PRESENTASI, False))


def terapkan_tema() -> None:
    """Suntikkan CSS tema (dan CSS Mode Presentasi bila aktif)."""
    css = _CSS_DASAR + (_CSS_PRESENTASI if presentasi_aktif() else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _ubah_presentasi() -> None:
    st.session_state[KUNCI_PRESENTASI] = not presentasi_aktif()


def tombol_presentasi_sidebar() -> None:
    """Tombol Mode Presentasi di sidebar (sidebar disembunyikan saat mode aktif)."""
    st.sidebar.button("Mode Presentasi", on_click=_ubah_presentasi, key="tombol_presentasi")


def judul(teks: str) -> None:
    """Siapkan tema, tombol Mode Presentasi di sidebar, dan judul halaman."""
    terapkan_tema()
    tombol_presentasi_sidebar()
    st.title(teks)


def footer() -> None:
    """Footer kecil; di Mode Presentasi juga memuat tombol untuk keluar dari mode."""
    st.markdown(f'<div class="footer-skripsi">{FOOTER}</div>', unsafe_allow_html=True)
    if presentasi_aktif():
        st.button("Keluar dari Mode Presentasi", on_click=_ubah_presentasi, key="keluar_presentasi")
