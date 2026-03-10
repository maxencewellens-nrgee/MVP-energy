NETWORK_TABLE = {
    (2026, "ORES", "BT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 65.0, "dso_fixe_eur_an": 120.0},
    (2026, "ORES", "MT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 30.0, "dso_fixe_eur_an": 600.0},
    (2026, "RESA", "BT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 60.0, "dso_fixe_eur_an": 150.0},
    (2026, "RESA", "MT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 28.0, "dso_fixe_eur_an": 620.0},
    (2026, "AIEG", "BT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 66.0, "dso_fixe_eur_an": 140.0},
    (2026, "AIEG", "MT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 32.0, "dso_fixe_eur_an": 680.0},
    (2026, "AIESH", "BT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 64.0, "dso_fixe_eur_an": 135.0},
    (2026, "AIESH", "MT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 31.0, "dso_fixe_eur_an": 665.0},
    (2026, "REW", "BT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 68.0, "dso_fixe_eur_an": 160.0},
    (2026, "REW", "MT"): {"transport_eur_mwh": 9.05, "dso_var_eur_mwh": 33.0, "dso_fixe_eur_an": 690.0},

    (2027, "ORES", "BT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 66.0, "dso_fixe_eur_an": 122.0},
    (2027, "ORES", "MT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 30.5, "dso_fixe_eur_an": 605.0},
    (2027, "RESA", "BT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 61.0, "dso_fixe_eur_an": 152.0},
    (2027, "RESA", "MT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 28.5, "dso_fixe_eur_an": 625.0},
    (2027, "AIEG", "BT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 67.0, "dso_fixe_eur_an": 142.0},
    (2027, "AIEG", "MT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 32.5, "dso_fixe_eur_an": 685.0},
    (2027, "AIESH", "BT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 65.0, "dso_fixe_eur_an": 137.0},
    (2027, "AIESH", "MT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 31.5, "dso_fixe_eur_an": 670.0},
    (2027, "REW", "BT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 69.0, "dso_fixe_eur_an": 162.0},
    (2027, "REW", "MT"): {"transport_eur_mwh": 9.10, "dso_var_eur_mwh": 33.5, "dso_fixe_eur_an": 695.0},

    (2028, "ORES", "BT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 64.0, "dso_fixe_eur_an": 121.0},
    (2028, "ORES", "MT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 29.8, "dso_fixe_eur_an": 602.0},
    (2028, "RESA", "BT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 59.0, "dso_fixe_eur_an": 151.0},
    (2028, "RESA", "MT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 28.0, "dso_fixe_eur_an": 623.0},
    (2028, "AIEG", "BT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 66.0, "dso_fixe_eur_an": 141.0},
    (2028, "AIEG", "MT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 32.0, "dso_fixe_eur_an": 682.0},
    (2028, "AIESH", "BT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 64.0, "dso_fixe_eur_an": 136.0},
    (2028, "AIESH", "MT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 31.0, "dso_fixe_eur_an": 668.0},
    (2028, "REW", "BT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 68.0, "dso_fixe_eur_an": 161.0},
    (2028, "REW", "MT"): {"transport_eur_mwh": 9.00, "dso_var_eur_mwh": 33.0, "dso_fixe_eur_an": 692.0},
}

def seg_code(label: str) -> str:
    """Convert segment label to code."""
    return "BT" if label.startswith("BT") else "MT"

def get_network_costs(annee: int, dso: str, seg_label: str) -> tuple:
    """Get network costs (transport, dso_var, dso_fixe)."""
    ref = NETWORK_TABLE.get((annee, dso, seg_code(seg_label)))
    if not ref:
        return 0.0, 0.0, 0.0
    return (
        float(ref["transport_eur_mwh"]),
        float(ref["dso_var_eur_mwh"]),
        float(ref["dso_fixe_eur_an"])
    )

def get_dsos_for_year(annee: int) -> list:
    """Get available DSOs for a year."""
    try:
        return sorted({d for (y, d, seg) in NETWORK_TABLE.keys() if y == annee})
    except:
        return []

def get_segments_for_dso(annee: int, dso: str) -> list:
    """Get available segments for a DSO in a year."""
    label_map = {"BT": "BT (≤56 kVA)", "MT": "MT (>56 kVA)"}
    try:
        segs = sorted({s for (y, dd, s) in NETWORK_TABLE.keys() if y == annee and dd == dso})
        return [label_map[s] for s in segs if s in label_map]
    except:
        return []
