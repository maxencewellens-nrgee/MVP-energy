import pandas as pd

NBSP = "\u00A0"

def _fmt_fr(val: float, dec: int = 2) -> str:
    """Format FR-BE: 98.560,50 (milliers='.', décimales=',')."""
    if val is None:
        val = 0.0
    s = f"{float(val):,.{dec}f}"
    s = s.replace(",", " ")
    s = s.replace(".", ",")
    s = s.replace(" ", ".")
    return s

def eur(val: float, dec: int = 2) -> str:
    """Format as EUR currency."""
    return f"{_fmt_fr(val, dec)}{NBSP}€"

def price_eur_mwh(p: float, dec: int = 2) -> str:
    """Format as EUR/MWh."""
    return f"{_fmt_fr(p, dec)}{NBSP}€/MWh"

def mwh(v: float, dec: int = 0) -> str:
    """Format as MWh."""
    return f"{_fmt_fr(v, dec)}{NBSP}MWh"

def fmt_be(d) -> str:
    """Format date as DD/MM/YYYY."""
    return pd.to_datetime(d).strftime("%d/%m/%Y")
