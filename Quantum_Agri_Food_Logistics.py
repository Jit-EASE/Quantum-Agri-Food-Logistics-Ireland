# quantum_agri_ireland_folium.py
# Streamlit app — Ireland choropleth (Folium) + slender arcs (flows)
# Real-world OSM A* routing + truck constraints + turn penalties
# QSTP (classical/quantum-inspired) + policy-aware costs + qLDPC stabilization
# + Monte Carlo + Markov + Difference-in-Differences (DiD) + GPT Decision Agent (robust API)
# Author: Jit | Updated: 2025-10-12

import os
import re
import math
import uuid
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from openai import OpenAI

# Folium stack
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

# Optional econometrics deps
try:
    import statsmodels.api as sm  # noqa: F401
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# Optional OSM real-world routing
try:
    import osmnx as ox
    HAS_OSMNX = True
except Exception:
    HAS_OSMNX = False

# ==============================
# Page + stable widget keys
# ==============================
st.set_page_config(page_title="Automated Agri Supply Chain Routes", layout="wide")

if "WKEY_NS" not in st.session_state:
    st.session_state["WKEY_NS"] = f"ns_{uuid.uuid4().hex[:8]}"

def wkey(name: str) -> str:
    return f"{st.session_state['WKEY_NS']}__{name}"

# ==============================
# Helpers
# ==============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ==============================
# Base data (synthetic nodes + county centroids)
# ==============================
COUNTY_COORDS = [
    ("Dublin", 53.3498, -6.2603),
    ("Cork", 51.8985, -8.4756),
    ("Galway", 53.2707, -9.0568),
    ("Limerick", 52.6638, -8.6267),
    ("Waterford", 52.2593, -7.1101),
    ("Kilkenny", 52.6541, -7.2448),
    ("Wexford", 52.3369, -6.4629),
    ("Kerry", 52.1545, -9.5669),
    ("Mayo", 53.8570, -9.2980),
    ("Donegal", 54.6538, -8.1096),
    ("Sligo", 54.2766, -8.4761),
    ("Westmeath", 53.5333, -7.3500),
    ("Tipperary", 52.4736, -8.1619),
]
COUNTY_DF = pd.DataFrame(COUNTY_COORDS, columns=["county", "lat", "lon"])

np.random.seed(42)
random.seed(42)

# Farms (from county centroids with jitter)
N_FARMS = 18
FARM_NODES: List[Dict] = []
for i in range(N_FARMS):
    base = COUNTY_DF.sample(1, random_state=42 + i).iloc[0]
    lat_jit = np.random.normal(0, 0.18)
    lon_jit = np.random.normal(0, 0.18)
    FARM_NODES.append({
        "id": f"farm_{i+1}",
        "type": "farm",
        "lat": float(base.lat + lat_jit),
        "lon": float(base.lon + lon_jit),
        "county": base.county,
    })

COOPS = [
    {"id": "coop_cork", "type": "coop", "lat": 51.901, "lon": -8.47, "county": "Cork"},
    {"id": "coop_galway", "type": "coop", "lat": 53.275, "lon": -9.06, "county": "Galway"},
    {"id": "coop_dublin", "type": "coop", "lat": 53.34, "lon": -6.26, "county": "Dublin"},
]
HUBS = [
    {"id": "hub_limerick", "type": "hub", "lat": 52.67, "lon": -8.62, "county": "Limerick"},
]

# Expanded ROI ports (commercial + key regional)
PORTS = [
    # East & East-North
    {"id": "port_dublin",        "type": "port", "lat": 53.346, "lon": -6.195, "name": "Dublin"},
    {"id": "port_dunlaoghaire",  "type": "port", "lat": 53.293, "lon": -6.135, "name": "Dún Laoghaire"},
    {"id": "port_howth",         "type": "port", "lat": 53.389, "lon": -6.065, "name": "Howth"},
    {"id": "port_drogheda",      "type": "port", "lat": 53.720, "lon": -6.250, "name": "Drogheda"},
    {"id": "port_dundalk",       "type": "port", "lat": 54.004, "lon": -6.404, "name": "Dundalk"},
    {"id": "port_greenore",      "type": "port", "lat": 54.039, "lon": -6.129, "name": "Greenore"},
    {"id": "port_wicklow",       "type": "port", "lat": 52.991, "lon": -6.033, "name": "Wicklow"},
    {"id": "port_arklow",        "type": "port", "lat": 52.795, "lon": -6.140, "name": "Arklow"},
    {"id": "port_rosslare",      "type": "port", "lat": 52.260, "lon": -6.340, "name": "Rosslare Europort"},

    # South & South-East
    {"id": "port_waterford",     "type": "port", "lat": 52.246, "lon": -7.031, "name": "Waterford (Belview)"},
    {"id": "port_newross",       "type": "port", "lat": 52.394, "lon": -6.951, "name": "New Ross"},
    {"id": "port_cork",          "type": "port", "lat": 51.860, "lon": -8.330, "name": "Port of Cork"},
    {"id": "port_kinsale",       "type": "port", "lat": 51.703, "lon": -8.526, "name": "Kinsale"},
    {"id": "port_bantry",        "type": "port", "lat": 51.656, "lon": -9.488, "name": "Bantry Bay / Whiddy"},

    # South-West & West
    {"id": "port_castletownbere", "type": "port", "lat": 51.650, "lon": -9.910, "name": "Castletownbere"},
    {"id": "port_dingle",         "type": "port", "lat": 52.140, "lon": -10.270, "name": "Dingle"},
    {"id": "port_fenit",          "type": "port", "lat": 52.280, "lon": -9.865, "name": "Fenit (Tralee)"},
    {"id": "port_shannon_foynes", "type": "port", "lat": 52.610, "lon": -9.110, "name": "Shannon Foynes"},
    {"id": "port_limerick",       "type": "port", "lat": 52.650, "lon": -8.650, "name": "Limerick Dock"},
    {"id": "port_galway",         "type": "port", "lat": 53.270, "lon": -9.050, "name": "Galway"},
    {"id": "port_rossaveal",      "type": "port", "lat": 53.266, "lon": -9.596, "name": "Rossaveal"},

    # North-West & North
    {"id": "port_sligo",         "type": "port", "lat": 54.280, "lon": -8.590, "name": "Sligo"},
    {"id": "port_killybegs",     "type": "port", "lat": 54.634, "lon": -8.452, "name": "Killybegs"},
    {"id": "port_donegal",       "type": "port", "lat": 54.628, "lon": -8.110, "name": "Donegal Bay (regional)"},
]

NODES = FARM_NODES + COOPS + HUBS + PORTS
NODES_DF = pd.DataFrame(NODES)

# ==============================
# Synthetic logistics graph + policy-aware costs
# ==============================
G = nx.Graph()
for n in NODES:
    G.add_node(n["id"], **n)
supply_targets = COOPS + HUBS

# Connect farms to nearest 2 coops/hubs
for farm in FARM_NODES:
    dists = []
    for t in supply_targets:
        d = haversine(farm["lat"], farm["lon"], t["lat"], t["lon"])
        dists.append((t["id"], d))
    dists.sort(key=lambda x: x[1])
    for t_id, d in dists[:2]:
        G.add_edge(farm["id"], t_id, distance=d, congestion=1.0, cost=d)

# Connect coops/hubs to ports
for a in supply_targets:
    for p in PORTS:
        d = haversine(a["lat"], a["lon"], p["lat"], p["lon"])
        congestion = np.random.uniform(0.9, 1.3)
        G.add_edge(a["id"], p["id"], distance=d, congestion=congestion, cost=d * congestion)

# Connect coops/hubs internally
for i in range(len(supply_targets)):
    for j in range(i + 1, len(supply_targets)):
        a, b = supply_targets[i], supply_targets[j]
        d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        G.add_edge(a["id"], b["id"], distance=d, congestion=1.05, cost=d * 1.05)

EMISSION_FACTOR_TONKM_DEFAULT = 0.0001

def edge_cost_km(distance_km: float, congestion: float, carbon_price: float,
                 load_tons: float = 1.0, gps_spreader: bool = False,
                 truck_co2_factor: float = EMISSION_FACTOR_TONKM_DEFAULT) -> float:
    base = distance_km * congestion
    emissions_t = distance_km * truck_co2_factor * load_tons
    carbon_cost = emissions_t * carbon_price
    rebate = 0.97 if gps_spreader else 1.0
    return (base + carbon_cost) * rebate

def apply_policy_costs(graph: nx.Graph, carbon_price: float, gps_spreader: bool, truck_co2_factor: float):
    for _, _, data in graph.edges(data=True):
        d = float(data.get("distance", 0.0))
        cong = float(data.get("congestion", 1.0))
        data["cost"] = edge_cost_km(d, cong, carbon_price, gps_spreader=gps_spreader,
                                    truck_co2_factor=truck_co2_factor)

# ==============================
# QSTP (classical + quantum-inspired) on synthetic graph
# ==============================
def routing_subgraph_for_target(G_full: nx.Graph, target_port_id: str, ports: List[Dict]) -> nx.Graph:
    """Copy G but remove all port nodes except the selected target (ports as sinks)."""
    H = G_full.copy()
    for p in ports:
        if p["id"] != target_port_id and H.has_node(p["id"]):
            H.remove_node(p["id"])
    return H

def classical_shortest_path(graph, source, target, weight="cost"):
    try:
        path = nx.shortest_path(graph, source=source, target=target, weight=weight)
        length = nx.path_weight(graph, path, weight)
        return path, length
    except nx.NetworkXNoPath:
        return [], float("inf")

def quantum_inspired_shortest_path(graph, source, target, weight="cost",
                                   n_trials: int = 64, temp: float = 1.0, seed: int = 123):
    random.seed(seed)
    best_path, best_len = classical_shortest_path(graph, source, target, weight)
    nodes = list(graph.nodes())
    for _ in range(n_trials):
        waypoint = random.choice(nodes)
        p1, l1 = classical_shortest_path(graph, source, waypoint, weight)
        p2, l2 = classical_shortest_path(graph, waypoint, target, weight)
        if not p1 or not p2:
            continue
        cand = p1[:-1] + p2
        clen = l1 + l2
        if clen < best_len:
            best_path, best_len = cand, clen
        else:
            prob = math.exp(-(clen - best_len) / max(1e-6, temp))
            if random.random() < prob:
                best_path, best_len = cand, clen
    return best_path, best_len

# ==============================
# qLDPC-style stabilization (ensemble + parity smoothing)
# ==============================
adj_map: Dict[str, List[str]] = {}
for c1, lat1, lon1 in COUNTY_COORDS:
    dists = []
    for c2, lat2, lon2 in COUNTY_COORDS:
        if c1 == c2:
            continue
        dists.append((c2, haversine(lat1, lon1, lat2, lon2)))
    dists.sort(key=lambda x: x[1])
    adj_map[c1] = [n for n, _ in dists[:3]]

def simulate_yield(counties: pd.DataFrame, drought: float, fert_cap: float,
                   carbon_price: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 6.0 + 0.6 * np.sin((counties.lat.values - 52) / 3)
    drought_penalty = drought * rng.normal(0.8, 0.1, len(counties))
    fert_penalty = fert_cap * rng.normal(0.5, 0.1, len(counties))
    carbon_bonus = (carbon_price / 100) * rng.normal(0.2, 0.05, len(counties))
    noise = rng.normal(0, 0.25, len(counties))
    y = base - drought_penalty - fert_penalty + carbon_bonus + noise
    y = np.clip(y, 1.0, None)
    return pd.DataFrame({"county": counties.county.values, "yield_t": y})

def qldpc_stabilize(counties: pd.DataFrame, drought: float, fert_cap: float, carbon_price: float,
                    ensemble: int = 21, apply_parity: bool = True, seed: int = 123) -> pd.DataFrame:
    runs = []
    for k in range(ensemble):
        runs.append(simulate_yield(counties, drought, fert_cap, carbon_price, seed=seed + k).set_index("county"))
    Y = pd.concat(runs, axis=1)
    Y.columns = [f"run_{i}" for i in range(ensemble)]
    med = Y.median(axis=1)
    df = med.to_frame("yield_t").reset_index()
    if apply_parity:
        tmp = df.set_index("county").copy()
        zt = 1.25
        for c in tmp.index:
            neigh = adj_map.get(c, [])
            if not neigh:
                continue
            vals = tmp.loc[neigh, "yield_t"].values
            mu = float(np.mean(vals))
            sd = float(np.std(vals)) or 1e-6
            z = (float(tmp.loc[c, "yield_t"]) - mu) / sd
            if abs(z) > zt:
                tmp.loc[c, "yield_t"] = 0.7 * float(tmp.loc[c, "yield_t"]) + 0.3 * mu
        df = tmp.reset_index()
    return df

# ==============================
# Econometrics: Monte Carlo, Markov chain, DiD
# ==============================
def monte_carlo_yields(
    counties_df: pd.DataFrame,
    drought: float, fert_cap: float, carbon_price: float,
    n_sims: int = 1000, seed: int = 123,
    nitrogen_limit: float = 0.0, multispecies_sward: bool = False
) -> pd.DataFrame:
    fert_scale = (1.0 - 0.4 * nitrogen_limit) * (0.9 if multispecies_sward else 1.0)
    sims = []
    for k in range(n_sims):
        yk = simulate_yield(
            counties_df,
            drought=drought,
            fert_cap=min(1.0, max(0.0, fert_cap * fert_scale)),
            carbon_price=carbon_price,
            seed=seed + k
        ).set_index("county")["yield_t"]
        sims.append(yk)
    Y = pd.concat(sims, axis=1)
    out = pd.DataFrame({
        "county": Y.index,
        "mean": Y.mean(axis=1).values,
        "sd": Y.std(axis=1).values,
        "p05": Y.quantile(0.05, axis=1).values,
        "p50": Y.quantile(0.50, axis=1).values,
        "p95": Y.quantile(0.95, axis=1).values,
    })
    return out.reset_index(drop=True)

def markov_weather_regimes(
    T: int,
    p_dd: float = 0.7,
    p_ww: float = 0.7,
    p0_dry: float = 0.5,
    dry_penalty: float = 0.5,
    wet_bonus: float = 0.2,
    seed: int = 123
) -> dict:
    transition = np.array([[p_dd, 1 - p_dd],
                           [1 - p_ww, p_ww]])
    denom = (2 - p_dd - p_ww)
    pi_dry = (1 - p_ww) / denom if denom != 0 else 0.5
    pi_wet = 1 - pi_dry
    stationary = np.array([pi_dry, pi_wet])

    rng = np.random.default_rng(seed)
    s = 0 if rng.uniform() < p0_dry else 1
    states = [s]
    for _ in range(T - 1):
        s = 0 if rng.uniform() < transition[s, 0] else 1
        states.append(s)

    return {"states": np.array(states), "transition": transition, "stationary": stationary,
            "dry_penalty": dry_penalty, "wet_bonus": wet_bonus}

def apply_weather_regime_to_yields(base_yield_df: pd.DataFrame, regime: dict, years: List[int]) -> pd.DataFrame:
    assert len(years) == len(regime["states"]), "Years must match Markov length"
    rows = []
    for i, yr in enumerate(years):
        adj = -regime["dry_penalty"] if regime["states"][i] == 0 else regime["wet_bonus"]
        yt = base_yield_df.copy()
        yt["yield_t"] = yt["yield_t"] + adj
        yt["year"] = yr
        rows.append(yt[["county", "year", "yield_t"]])
    return pd.concat(rows, axis=0).reset_index(drop=True)

def difference_in_differences(
    panel_df: pd.DataFrame,
    treated_counties: List[str],
    treat_year: int,
    cluster_se: bool = True
) -> dict:
    panel = panel_df.copy()
    panel["treated"] = panel["county"].isin(treated_counties).astype(int)
    panel["post"] = (panel["year"] >= treat_year).astype(int)
    panel["did"] = panel["treated"] * panel["post"]

    if HAS_STATSMODELS:
        model = smf.ols("yield_t ~ treated*post + C(county) + C(year)", data=panel).fit(
            cov_type=("cluster" if cluster_se else "nonrobust"),
            cov_kwds=({"groups": panel["county"]} if cluster_se else None)
        )
        att = model.params.get("treated:post", np.nan)
        se = model.bse.get("treated:post", np.nan)
        pval = model.pvalues.get("treated:post", np.nan)
        return {"model": model, "att": att, "se": se, "pvalue": pval, "nobs": model.nobs,
                "note": "TWFE (statsmodels)"}

    pre = panel[panel["year"] < treat_year]
    post = panel[panel["year"] >= treat_year]
    d_treated = post[post["treated"] == 1]["yield_t"].mean() - pre[pre["treated"] == 1]["yield_t"].mean()
    d_control = post[post["treated"] == 0]["yield_t"].mean() - pre[pre["treated"] == 0]["yield_t"].mean()
    att = d_treated - d_control
    return {"model": None, "att": att, "se": np.nan, "pvalue": np.nan, "nobs": len(panel),
            "note": "2x2 DiD (fallback)"}

# ==============================
# Real-world routing (OSM A*)
# ==============================
def build_osm_drive_graph(place: str = "Ireland",
                          truck_gvw_tons: Optional[float] = None,
                          min_speed_kph: float = 5.0) -> Optional["nx.MultiDiGraph"]:
    if not HAS_OSMNX:
        return None
    ox.settings.use_cache = True
    ox.settings.log_console = False

    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    # speeds & travel time
    try:
        G = ox.add_edge_speeds(G)          # older/newer API variant
    except Exception:
        pass
    try:
        G = ox.add_edge_travel_times(G)
    except Exception:
        # fallback: approximate from length & speed_kph if missing
        for _, _, k, data in G.edges(keys=True, data=True):
            sp = data.get("speed_kph", 50.0) or 50.0
            dist_km = (data.get("length", 0.0) or 0.0)/1000.0
            data["travel_time"] = (dist_km / max(5.0, sp)) * 3600.0

    # bearings for turn costs
    try:
        G = ox.bearing.add_edge_bearings(G)
    except Exception:
        try:
            G = ox.add_edge_bearings(G)
        except Exception:
            pass

    # coarse truck filtering
    for u, v, k, data in list(G.edges(keys=True, data=True)):
        sp = data.get("speed_kph", None)
        if sp is None or sp <= 0:
            data["speed_kph"] = max(min_speed_kph, 30.0)

        if truck_gvw_tons is not None:
            hgv = str(data.get("hgv", "")).lower()
            if hgv in ("no", "destination"):
                G.remove_edge(u, v, k)
                continue
            mw = data.get("maxweight")
            if mw is not None:
                try:
                    m = re.search(r"[0-9]+(\.[0-9]+)?", str(mw))
                    if m and float(m.group()) < float(truck_gvw_tons):
                        G.remove_edge(u, v, k)
                        continue
                except Exception:
                    pass
    return G

def _edge_cost_hours(data: dict,
                     w_time: float = 1.0,
                     w_dist: float = 0.0,
                     w_emiss: float = 0.0,
                     truck_load_tons: float = 1.0,
                     co2_per_ton_km: float = 0.0001) -> float:
    dist_km = (data.get("length", 0.0) or 0.0) / 1000.0
    tt_h = ((data.get("travel_time", None) or (dist_km / max(5.0, data.get("speed_kph", 50.0)) * 3600.0)) / 3600.0)
    emis = dist_km * co2_per_ton_km * truck_load_tons
    return w_time * tt_h + w_dist * dist_km + w_emiss * emis

def apply_osm_cost(G: "nx.MultiDiGraph",
                   w_time: float, w_dist: float, w_emiss: float,
                   truck_load_tons: float, co2_per_ton_km: float) -> None:
    for u, v, k, data in G.edges(keys=True, data=True):
        data["gen_cost"] = _edge_cost_hours(
            data, w_time=w_time, w_dist=w_dist, w_emiss=w_emiss,
            truck_load_tons=truck_load_tons, co2_per_ton_km=co2_per_ton_km
        )

def _bearing_on(G, u, v, k) -> Optional[float]:
    try:
        return G[u][v][k].get("bearing")
    except Exception:
        return None

def approximate_turn_penalty_seconds(G: "nx.MultiDiGraph", path: List[int]) -> float:
    if len(path) < 3:
        return 0.0
    penalty_s = 0.0
    for i in range(len(path) - 2):
        u, v, w = path[i], path[i+1], path[i+2]
        try:
            k1 = next(iter(G[u][v].keys()))
            k2 = next(iter(G[v][w].keys()))
        except Exception:
            continue
        b1 = _bearing_on(G, u, v, k1)
        b2 = _bearing_on(G, v, w, k2)
        if b1 is None or b2 is None:
            continue
        ang = abs((b2 - b1 + 180) % 360 - 180)  # [0,180]
        if ang > 170:
            penalty_s += 60.0  # strong U-turn discouragement
        penalty_s += (ang * ang) / 800.0  # ~40s @180°, ~5s @63°
    return penalty_s

def osm_astar_route(G: "nx.MultiDiGraph",
                    orig_lat: float, orig_lon: float,
                    dest_lat: float, dest_lon: float,
                    w_time: float = 1.0, w_dist: float = 0.0, w_emiss: float = 0.0,
                    truck_load_tons: float = 1.0, co2_per_ton_km: float = 0.0001):
    assert HAS_OSMNX, "OSM routing requires osmnx."
    orig = ox.nearest_nodes(G, X=orig_lon, Y=orig_lat)
    dest = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)
    apply_osm_cost(G, w_time, w_dist, w_emiss, truck_load_tons, co2_per_ton_km)

    def h(n1, n2):
        y1, x1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
        y2, x2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
        d = haversine(y1, x1, y2, x2)  # km
        return d / 110.0  # optimistic hours @ 110 kph

    path = nx.astar_path(G, orig, dest, heuristic=h, weight="gen_cost", allow_switches=True)
    length_h = nx.path_weight(G, path, weight="gen_cost")  # generalized-hours
    turns_s = approximate_turn_penalty_seconds(G, path)
    return path, length_h, turns_s

# ==============================
# Folium builder: choropleth + slender arcs + highlighted route
# ==============================
def build_ireland_folium_map(
    yield_df: pd.DataFrame,
    gj: dict | None = None,
    flows_df: pd.DataFrame | None = None,
    route_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] | None = None,
    county_key_candidates: tuple[str, ...] = ("county", "name", "NAME"),
    initial_center: tuple[float, float] = (53.4, -8.0),
    zoom_start: int = 6,
) -> folium.Map:
    m = folium.Map(location=initial_center, zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True)

    # Choropleth
    if gj is not None:
        by_name = {str(c).lower(): float(v) for c, v in zip(yield_df["county"], yield_df["yield_t"])}

        def _county_name(props: dict) -> Optional[str]:
            for k in county_key_candidates:
                if k in props and props[k]:
                    return str(props[k]).lower()
            return None

        y_min = float(yield_df["yield_t"].min())
        y_max = float(yield_df["yield_t"].max())
        cmap = LinearColormap(colors=["#3f8efc", "#5cb85c", "#f0ad4e", "#d9534f"], vmin=y_min, vmax=y_max)
        cmap.caption = "Yield (t/ha)"

        def style_fn(feature):
            props = feature.get("properties", {})
            name = _county_name(props)
            yv = by_name.get(name, None)
            fill = "#dddddd" if yv is None else cmap(yv)
            return {"fillColor": fill, "color": "#666666", "weight": 0.8, "fillOpacity": 0.85 if yv is not None else 0.5}

        folium.GeoJson(
            gj,
            style_function=style_fn,
            highlight_function=lambda x: {"weight": 2, "color": "#333333"},
            name="Yield choropleth",
        ).add_to(m)

        # Popups
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            name = props.get("county") or props.get("name") or props.get("NAME") or "Unknown county"
            yv = by_name.get(str(name).lower(), None)
            val = "N/A" if yv is None else f"{yv:.2f} t/ha"
            folium.GeoJson(
                feat,
                style_function=lambda x: {"weight": 0, "opacity": 0},
                popup=folium.Popup(f"<b>{name}</b><br>Yield: {val}", max_width=260),
            ).add_to(m)

        cmap.add_to(m)

    else:
        # Fallback: centroid markers
        y_min = float(yield_df["yield_t"].min()); y_max = float(yield_df["yield_t"].max())
        cmap = LinearColormap(["#3f8efc", "#5cb85c", "#f0ad4e", "#d9534f"], vmin=y_min, vmax=y_max)
        cmap.caption = "Yield (t/ha)"
        tmp = COUNTY_DF.merge(yield_df, on="county", how="left")
        for _, r in tmp.iterrows():
            color = "#cccccc" if pd.isna(r["yield_t"]) else cmap(float(r["yield_t"]))
            folium.CircleMarker(
                location=(r["lat"], r["lon"]),
                radius=7,
                color="#444444",
                weight=0.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"{r['county']}: {float(r['yield_t']):.2f} t/ha" if not pd.isna(r["yield_t"]) else f"{r['county']}: N/A",
            ).add_to(m)
        cmap.add_to(m)

    # Slender arcs (flows)
    if flows_df is not None and not flows_df.empty:
        for _, r in flows_df.iterrows():
            src = (float(r["source_lat"]), float(r["source_lon"]))
            dst = (float(r["target_lat"]), float(r["target_lon"]))
            vol = float(r.get("volume", 1.0))
            folium.PolyLine(
                locations=[src, dst],
                color="#111111",
                weight=max(1.0, min(3.0, 0.02 * vol)),
                opacity=0.5,
                tooltip=f"Volume: {vol:.0f}",
            ).add_to(m)

    # Highlighted route
    if route_segments:
        for (lat1, lon1), (lat2, lon2) in route_segments:
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color="#000000",
                weight=4.0,
                opacity=0.9,
            ).add_to(m)

    # Fit bounds
    try:
        if gj is not None and "features" in gj and gj["features"]:
            lats, lons = [], []
            for feat in gj["features"]:
                geom = feat.get("geometry")
                if not geom: continue
                coords = []
                if geom["type"] == "Polygon":
                    for ring in geom["coordinates"]:
                        coords.extend(ring)
                elif geom["type"] == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        for ring in poly:
                            coords.extend(ring)
                for lon, lat in coords:
                    lats.append(lat); lons.append(lon)
            if lats and lons:
                m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        else:
            m.fit_bounds([[51.3, -10.7], [55.5, -5.3]])
    except Exception:
        pass

    folium.LayerControl(collapsed=True).add_to(m)
    return m

# ==============================
# Sidebar controls
# ==============================
with st.sidebar:
    st.header("Scenario Controls")

    st.markdown("**Climate & Policy Inputs**")
    drought = st.slider("Drought severity", 0.0, 1.0, 0.35, 0.05, key=wkey("drought"))
    fert_cap = st.slider("Fertilizer cap (relative)", 0.0, 1.0, 0.30, 0.05, key=wkey("fert"))
    carbon_price = st.slider("Carbon price (€/tCO₂e)", 0.0, 200.0, 80.0, 5.0, key=wkey("carbon"))
    truck_co2_factor = st.number_input("Truck CO₂ factor (tCO₂/ton-km)", min_value=0.0, value=0.0001,
                                       step=0.00001, format="%.5f", key=wkey("co2_factor"))

    st.markdown("**Eco-scheme controls**")
    nitrogen_limit = st.slider("Nitrogen limit intensity", 0.0, 1.0, 0.4, 0.05, key=wkey("nitrogen"))
    gps_spreader = st.checkbox("GPS-controlled spreading", value=True, key=wkey("gps"))
    multispecies_sward = st.checkbox("Multispecies sward adoption", value=False, key=wkey("sward"))
    adoption_rate = st.slider("Eco-scheme adoption %", 0, 100, 40, 5, key=wkey("adoption"))
    nitrates_per_ha = st.number_input("Nitrate use (kg/ha)", min_value=0.0, value=150.0, step=5.0, key=wkey("nitrates"))

    st.markdown("**Time**")
    year = st.slider("Year", 2015, 2035, 2025, 1, key=wkey("year"))

    st.markdown("---")
    st.markdown("**Routing network source**")
    routing_source = st.radio(
        "Choose network",
        options=["OSM (real-world A*)", "Synthetic (demo graph)"],
        index=0,
        key=wkey("routing_source")
    )
    truck_gvw = st.number_input("Truck gross weight (t)", min_value=1.0, value=26.0, step=1.0, key=wkey("truck_gvw"))
    truck_load_tons = st.number_input("Payload (t)", min_value=0.1, value=20.0, step=0.5, key=wkey("truck_payload"))
    w_time = st.slider("Cost weight — time", 0.0, 2.0, 1.0, 0.1, key=wkey("w_time"))
    w_dist = st.slider("Cost weight — distance", 0.0, 2.0, 0.0, 0.1, key=wkey("w_dist"))
    w_emiss = st.slider("Cost weight — emissions", 0.0, 2.0, 0.5, 0.1, key=wkey("w_emiss"))

    st.markdown("---")
    st.markdown("**Routing (QSTP)**")
    solver = st.selectbox("Routing solver (synthetic only)", ["Classical (Dijkstra)", "Quantum-inspired (annealing)"], key=wkey("solver"))
    n_trials = st.slider("Annealing trials", 8, 256, 64, 8, key=wkey("trials"))
    temp = st.slider("Annealing temperature", 0.1, 5.0, 1.0, 0.1, key=wkey("temp"))

    st.markdown("**qLDPC Stabilizer**")
    use_qldpc = st.checkbox("Enable ensemble + parity smoothing", value=True, key=wkey("qldpc"))
    ensemble = st.slider("Ensemble size (runs)", 5, 51, 21, 2, key=wkey("ensemble"))

    st.markdown("---")
    st.markdown("**GeoJSON (optional)**")
    gj_file = st.file_uploader("Upload county GeoJSON (must contain county name property)",
                               type=["json", "geojson"], key=wkey("gj_upl"))

    st.markdown("---")
    st.markdown("**Econometrics**")
    enable_mc = st.checkbox("Monte Carlo (yield uncertainty)", value=True, key=wkey("mc_enable"))
    n_sims = st.slider("MC simulations", 100, 5000, 1000, 100, key=wkey("mc_sims"))

    enable_markov = st.checkbox("Markov chain (weather regimes)", value=False, key=wkey("mkv_enable"))
    p_dd = st.slider("P(dry→dry)", 0.0, 1.0, 0.7, 0.05, key=wkey("p_dd"))
    p_ww = st.slider("P(wet→wet)", 0.0, 1.0, 0.7, 0.05, key=wkey("p_ww"))
    dry_penalty = st.number_input("Dry penalty (t/ha)", value=0.5, step=0.1, key=wkey("dry_pen"))
    wet_bonus = st.number_input("Wet bonus (t/ha)", value=0.2, step=0.1, key=wkey("wet_bonus"))
    panel_start = st.number_input("Panel start year", value=2015, step=1, key=wkey("panel_start"))
    panel_end = st.number_input("Panel end year", value=2035, step=1, key=wkey("panel_end"))

    enable_did = st.checkbox("Difference-in-Differences (DiD)", value=False, key=wkey("did_enable"))
    treat_year = st.number_input("Treatment year", value=2023, step=1, key=wkey("treat_year"))
    treated_counties = st.multiselect("Treated counties",
                                      options=COUNTY_DF["county"].tolist(),
                                      default=["Dublin", "Cork"], key=wkey("treated_cs"))
    cluster_se = st.checkbox("Cluster SEs by county (requires statsmodels)", value=True, key=wkey("did_cluster"))

    st.markdown("---")
    st.markdown("**AI Decision Agent**")
    use_agent = st.checkbox("Enable GPT agent for route decision", value=True, key=wkey("agent_enable"))
    model_choice = st.selectbox("Agent model", options=["gpt-4.1-mini", "gpt-4o-mini"], index=0,
                                key=wkey("agent_model"), help="Use 4.1 mini if available; otherwise 4o-mini.")

    st.markdown("---")
    st.markdown("**Presets**")
    if st.button("Save current preset", key=wkey("save_preset")):
        st.session_state["preset"] = {
            "drought": drought, "fert_cap": fert_cap, "carbon_price": carbon_price,
            "nitrogen_limit": nitrogen_limit, "gps_spreader": gps_spreader, "multispecies_sward": multispecies_sward,
            "adoption_rate": adoption_rate, "nitrates_per_ha": nitrates_per_ha,
            "year": year, "solver": solver, "n_trials": n_trials, "temp": temp,
            "use_qldpc": use_qldpc, "ensemble": ensemble, "truck_co2_factor": truck_co2_factor,
            "enable_mc": enable_mc, "n_sims": int(n_sims),
            "enable_markov": enable_markov, "p_dd": float(p_dd), "p_ww": float(p_ww),
            "dry_penalty": float(dry_penalty), "wet_bonus": float(wet_bonus),
            "panel_start": int(panel_start), "panel_end": int(panel_end),
            "enable_did": enable_did, "treat_year": int(treat_year),
            "treated_counties": treated_counties, "cluster_se": bool(cluster_se),
            "use_agent": use_agent, "model_choice": model_choice,
            "routing_source": routing_source, "truck_gvw": float(truck_gvw),
            "truck_load_tons": float(truck_load_tons),
            "w_time": float(w_time), "w_dist": float(w_dist), "w_emiss": float(w_emiss)
        }
        st.success("Preset saved in session.")
    if st.button("Load preset", key=wkey("load_preset")):
        p = st.session_state.get("preset")
        if p:
            # restore (widgets will rerender on rerun)
            for k, v in p.items():
                st.session_state[wkey(k)] = v
            st.experimental_rerun()
        else:
            st.info("No preset stored yet.")

# ==============================
# Main UI
# ==============================
st.title("Automated AgriChain Network (Fusion Econometric System) — Ireland")
st.caption("OSM A* routing • Ports as sinks • Choropleth + arcs • QSTP • qLDPC • MC • Markov • DiD • GPT agent")

# Apply policy costs (synthetic layer)
apply_policy_costs(G, carbon_price=carbon_price, gps_spreader=gps_spreader, truck_co2_factor=truck_co2_factor)

# A) Routing & Flows
st.markdown("### A. Routing & Flows")
farm_ids = [n["id"] for n in FARM_NODES]

# Friendly port dropdown -> internal id
port_options = {p.get("name", p["id"]): p["id"] for p in PORTS}
port_label_list = list(port_options.keys())

c1, c2, c3 = st.columns([1,1,1])
with c1:
    source = st.selectbox("Select source farm", farm_ids, index=0, key=wkey("src"))
with c2:
    target_label = st.selectbox("Select export port", port_label_list, index=0, key=wkey("tgt_label"))
    target = port_options[target_label]
with c3:
    st.caption("Arcs show co-op/hub → selected port flows.")

# Build synthetic flows coop/hub -> selected port (for map arcs)
ch_nodes = COOPS + HUBS
farm_nearest = {c["id"]: 0 for c in ch_nodes}
for farm in FARM_NODES:
    dlist = [(c["id"], haversine(farm["lat"], farm["lon"], c["lat"], c["lon"])) for c in ch_nodes]
    dlist.sort(key=lambda x: x[1])
    farm_nearest[dlist[0][0]] += 1

year_factor = 1.0 + 0.01 * (year - 2020)
flows = []
port_node = next(p for p in PORTS if p["id"] == target)
for c in ch_nodes:
    base_vol = max(10, 12 * farm_nearest[c["id"]])
    vol = int(base_vol * year_factor)
    flows.append({
        "source_lat": c["lat"], "source_lon": c["lon"],
        "target_lat": port_node["lat"], "target_lon": port_node["lon"],
        "volume": vol
    })
flow_df = pd.DataFrame(flows)

# ---------------- OSM Graph Build/Cache ----------------
OSM_G = None
if routing_source.startswith("OSM"):
    if not HAS_OSMNX:
        st.error("OSM routing selected but `osmnx` is not installed. `pip install osmnx`")
    else:
        if "OSM_G" not in st.session_state:
            with st.spinner("Loading OSM road network for Ireland…"):
                st.session_state["OSM_G"] = build_osm_drive_graph("Ireland", truck_gvw_tons=float(truck_gvw))
        OSM_G = st.session_state.get("OSM_G")

# ---------------- Solve route ----------------
route_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
path, plen = [], float("inf")
route_hours, dist_km, turn_pen_s = 0.0, 0.0, 0.0

if routing_source.startswith("OSM") and HAS_OSMNX and OSM_G is not None:
    farm_node = next(n for n in NODES if n["id"] == source)
    try:
        node_path, plen_h, turn_s = osm_astar_route(
            OSM_G,
            farm_node["lat"], farm_node["lon"],
            port_node["lat"], port_node["lon"],
            w_time=w_time, w_dist=w_dist, w_emiss=w_emiss,
            truck_load_tons=float(truck_load_tons),
            co2_per_ton_km=float(truck_co2_factor)
        )
        # Convert node path to segments for Folium
        for u, v in zip(node_path[:-1], node_path[1:]):
            route_segments.append(((OSM_G.nodes[u]["y"], OSM_G.nodes[u]["x"]),
                                   (OSM_G.nodes[v]["y"], OSM_G.nodes[v]["x"])))
        # KPIs
        route_hours = float(plen_h)
        try:
            # sum length across first parallel edge (approx)
            for u, v in zip(node_path[:-1], node_path[1:]):
                k = next(iter(OSM_G[u][v].keys()))
                dist_km += (OSM_G[u][v][k].get("length", 0.0) or 0.0) / 1000.0
        except Exception:
            pass
        turn_pen_s = float(turn_s)
        path = [source, port_node["id"]]
        plen = dist_km
        st.info(f"OSM A* route ≈ {route_hours:.2f} hours (+~{turn_pen_s:.0f}s turns), distance ≈ {dist_km:.1f} km.")
    except Exception as e:
        st.error(f"OSM routing failed: {e}")
        path, plen, route_segments = [], float("inf"), []
else:
    # Synthetic graph with ports-as-sinks subgraph
    H = routing_subgraph_for_target(G, target, PORTS)
    if solver.startswith("Classical"):
        path, plen = classical_shortest_path(H, source, target, weight="cost")
    else:
        path, plen = quantum_inspired_shortest_path(H, source, target, weight="cost", n_trials=int(n_trials), temp=float(temp))
    if len(path) >= 2:
        for u, v in zip(path[:-1], path[1:]):
            u_node, v_node = H.nodes[u], H.nodes[v]
            route_segments.append(((u_node["lat"], u_node["lon"]), (v_node["lat"], v_node["lon"])))
    # Synthetic KPIs (distance-based)
    dist_km = float(plen) if np.isfinite(plen) else 0.0
    route_hours = dist_km/60.0 if dist_km > 0 else 0.0
    st.info(f"Synthetic {solver} path ≈ {dist_km:.1f} km (~{route_hours:.2f} h @60kph). Ports are sinks (no transit).")

if path:
    st.code(" -> ".join(path), language="text")
else:
    st.warning("No path found with current configuration.")

st.markdown("---")
st.markdown("### B. Supply Network Map — Ireland’s Agri-Food Logistics and Yield Dynamics")

# Yields (year-modulated & qLDPC-style)
yr_phase = (year - 2015) / 20.0
adj_drought = float(min(1.0, max(0.0, drought + 0.15 * math.sin(2 * math.pi * yr_phase))))
fert_scale = (1.0 - 0.4 * nitrogen_limit) * (0.9 if multispecies_sward else 1.0)
adj_fert = float(min(1.0, max(0.0, fert_cap * fert_scale + 0.10 * math.cos(2 * math.pi * yr_phase))))
adj_carbon = float(max(0.0, carbon_price * (0.9 + 0.2 * yr_phase)))

if use_qldpc:
    yield_df = qldpc_stabilize(COUNTY_DF, drought=adj_drought, fert_cap=adj_fert, carbon_price=adj_carbon, ensemble=int(ensemble))
    st.caption(f"{year} • Stabilized via ensemble={ensemble} + parity smoothing")
else:
    yield_df = simulate_yield(COUNTY_DF, drought=adj_drought, fert_cap=adj_fert, carbon_price=adj_carbon, seed=7)
    st.caption(f"{year} • Single-run (no stabilization)")

# GeoJSON (optional)
_gj = None
if gj_file is not None:
    try:
        import json
        _gj = json.loads(gj_file.read())
    except Exception as e:
        st.warning(f"Failed to read GeoJSON: {e}")

# Render Folium map
ireland_map = build_ireland_folium_map(
    yield_df=yield_df,
    gj=_gj,
    flows_df=flow_df,
    route_segments=route_segments
)
st_folium(ireland_map, width=None, height=700)

with st.expander("Show county yield table"):
    st.dataframe(yield_df.sort_values("yield_t", ascending=False).reset_index(drop=True))

st.markdown("---")
st.markdown("### C. SDG/ESG KPIs")
mean_yield = float(yield_df['yield_t'].mean())
route_emissions_t = float(dist_km) * float(truck_co2_factor)  # tCO2 per payload-ton
colA, colB, colC, colD = st.columns(4)
colA.metric("SDG 2 — Yield adequacy (t/ha)", f"{mean_yield:.2f}")
colB.metric("SDG 12 — Nitrates (kg/ha)", f"{nitrates_per_ha:.0f}")
colC.metric("SDG 13 — Route emissions (tCO₂ per ton)", f"{route_emissions_t:.4f}")
colD.metric("Travel time (hours)", f"{route_hours:.2f}")

# ==============================
# D. Monte Carlo — yield uncertainty
# ==============================
if enable_mc:
    st.markdown("### D. Monte Carlo — Yield Uncertainty")
    mc_stats = monte_carlo_yields(
        COUNTY_DF,
        drought=adj_drought, fert_cap=adj_fert, carbon_price=adj_carbon,
        n_sims=int(n_sims), seed=123,
        nitrogen_limit=nitrogen_limit, multispecies_sward=multispecies_sward
    )
    st.caption(f"{n_sims} simulations • per-county mean/sd and 5th/50th/95th percentiles")
    st.dataframe(mc_stats.sort_values("mean", ascending=False).reset_index(drop=True))

    hi_risk = mc_stats.sort_values("sd", ascending=False).head(5)[["county", "sd", "p05", "p95"]]
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Top 5 counties by uncertainty (sd)**")
        st.dataframe(hi_risk.reset_index(drop=True))
    with c2:
        st.metric("Portfolio CV (avg sd/mean)", f"{(mc_stats['sd']/mc_stats['mean']).mean():.3f}")

# ==============================
# E. Markov chain — weather regimes
# ==============================
panel_yields = None
if enable_markov:
    st.markdown("### E. Markov Chain — Weather Regimes")
    years_panel = list(range(int(panel_start), int(panel_end) + 1))
    T = len(years_panel)

    regime = markov_weather_regimes(
        T=T, p_dd=float(p_dd), p_ww=float(p_ww), p0_dry=0.5,
        dry_penalty=float(dry_penalty), wet_bonus=float(wet_bonus), seed=123
    )
    st.caption(
        f"Stationary — Dry: {regime['stationary'][0]:.2f}, Wet: {regime['stationary'][1]:.2f} | "
        f"Realized Dry share: {(regime['states']==0).mean():.2f}"
    )
    panel_yields = apply_weather_regime_to_yields(yield_df, regime, years_panel)
    with st.expander("Show Markov panel yields (head)"):
        st.dataframe(panel_yields.head(20))

    panel_summary = panel_yields.groupby("county")["yield_t"].agg(["mean", "std"]).reset_index().rename(columns={"std": "sd"})
    st.dataframe(panel_summary.sort_values("mean", ascending=False))

# ==============================
# F. Difference-in-Differences (DiD)
# ==============================
if enable_did:
    st.markdown("### F. Difference-in-Differences (DiD) — Eco-scheme impact")
    if panel_yields is not None:
        did_panel = panel_yields.copy()
    else:
        years_panel = list(range(int(panel_start), int(panel_end) + 1))
        rows = []
        for yr in years_panel:
            tmp = yield_df.copy()
            tmp["year"] = yr
            rows.append(tmp[["county", "year", "yield_t"]])
        did_panel = pd.concat(rows, axis=0).reset_index(drop=True)

    result = difference_in_differences(
        panel_df=did_panel,
        treated_counties=treated_counties,
        treat_year=int(treat_year),
        cluster_se=bool(cluster_se)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATT (treated×post)", f"{result['att']:.3f}")
    col2.metric("SE", "n/a" if np.isnan(result["se"]) else f"{result['se']:.3f}")
    col3.metric("p-value", "n/a" if np.isnan(result["pvalue"]) else f"{result['pvalue']:.3f}")
    col4.metric("Obs", f"{int(result['nobs'])}")
    st.caption(f"Estimator: {result['note']} (statsmodels available={HAS_STATSMODELS})")

    if HAS_STATSMODELS and result["model"] is not None:
        with st.expander("Show DiD regression summary"):
            st.text(result["model"].summary())

# ==============================
# GPT Agent (robust OpenAI call)
# ==============================
def build_agent_context(
    year: int,
    solver: str,
    path: List[str],
    plen_km: float,
    gps_spreader: bool,
    truck_co2_factor: float,
    yield_df: pd.DataFrame,
    mc_stats: Optional[pd.DataFrame],
    regime: Optional[dict],
    did_result: Optional[dict],
    flow_df: Optional[pd.DataFrame],
    carbon_price: float,
    nitrates_per_ha: float,
    adoption_rate: int
) -> dict:
    ctx = {
        "year": year,
        "solver": solver,
        "route": {
            "nodes": path,
            "length_km": None if (plen_km is None or (isinstance(plen_km, float) and not np.isfinite(plen_km))) else float(plen_km),
            "gps_spreader": bool(gps_spreader),
            "truck_co2_factor_t_per_ton_km": float(truck_co2_factor),
            "est_route_emissions_t_per_ton": (0.0 if plen_km in [None, float('inf')] else float(plen_km) * float(truck_co2_factor))
        },
        "policy": {
            "carbon_price_eur_per_t": float(carbon_price),
            "nitrates_kg_per_ha": float(nitrates_per_ha),
            "eco_scheme_adoption_pct": int(adoption_rate),
        },
        "yields": {
            "by_county": yield_df.sort_values("county").to_dict(orient="records"),
            "mean": float(yield_df["yield_t"].mean()),
            "min": float(yield_df["yield_t"].min()),
            "max": float(yield_df["yield_t"].max()),
        },
        "flows_sample": (None if flow_df is None else flow_df.head(25).to_dict(orient="records")),
        "uncertainty_mc": (None if mc_stats is None else {
            "by_county": mc_stats.sort_values("county").to_dict(orient="records"),
            "portfolio_cv": float((mc_stats["sd"]/mc_stats["mean"]).mean())
        }),
        "markov": (None if regime is None else {
            "stationary": list(map(float, regime["stationary"])),
            "realized_dry_share": float((regime["states"]==0).mean()),
            "dry_penalty": float(regime["dry_penalty"]),
            "wet_bonus": float(regime["wet_bonus"])
        }),
        "did": (None if did_result is None else {
            "att": None if did_result.get("att") is None else float(did_result["att"]),
            "se": None if np.isnan(did_result.get("se", np.nan)) else float(did_result["se"]),
            "pvalue": None if np.isnan(did_result.get("pvalue", np.nan)) else float(did_result["pvalue"]),
            "note": did_result.get("note")
        })
    }
    return ctx

def _parse_json_loose(text: str) -> dict:
    import json as _json
    if not isinstance(text, str):
        raise ValueError("No text to parse from model output.")
    try:
        return _json.loads(text)
    except Exception:
        pass
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return _json.loads(candidate)
        except Exception:
            pass
    raise ValueError("Model did not return valid JSON.")

def call_openai_agent(model: str, context: dict) -> dict:
    import json as _json
    client = OpenAI()  # reads OPENAI_API_KEY
    schema_keys_hint = (
        "Return ONLY a JSON object with keys: "
        "route_verdict, primary_objective, rationale, risk_flags, sensitivity, recommended_actions, kpis. "
        "Within sensitivity include carbon_price_breakpoint, nitrate_threshold_kg_ha, mc_p95_drop_t_ha. "
        "Within kpis include est_route_len_km, est_emissions_t_per_ton, mean_yield_t_ha, portfolio_cv."
    )
    system_msg = (
        "You are a supply chain policy co-pilot for Irish agriculture. "
        "Given routing (QSTP/OSM), stabilized yields (qLDPC), Markov weather regimes, Monte Carlo uncertainty, "
        "and a DiD policy effect, decide whether to accept the proposed route, or how to revise it. "
        "Be conservative when uncertainty or DiD indicates harm. "
        "Output must be strict JSON (no prose outside JSON)."
    )
    user_msg = (
        "Objectives, in order: minimize emissions subject to acceptable cost and resilience. "
        "Consider eco-scheme adoption, nitrates limits, and high-uncertainty counties. " + schema_keys_hint
    )
    # 1) Responses API (without response_format)
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": {"type": "input_json", "json": context}},
            ],
        )
        out = getattr(resp, "output_json", None)
        if isinstance(out, dict):
            return out
        try:
            text = getattr(resp, "output_text", None)
            if text is None:
                text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)
        return _parse_json_loose(text)
    except TypeError:
        pass
    except Exception:
        pass
    # 2) Chat Completions fallback (JSON mode)
    cc = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "user", "content": f"Context JSON:\n{_parse_json_loose if False else _json.dumps(context)}"},
        ],
    )
    text = cc.choices[0].message.content
    return _parse_json_loose(text)

# Safety defaults for agent variables
try:
    use_agent
except NameError:
    use_agent = False
try:
    model_choice
except NameError:
    model_choice = "gpt-4o-mini"

if use_agent and not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. The GPT agent will fail to run. Set it in your environment.")

# ==============================
# G. GPT Agent — contextual decision for optimal routing
# ==============================
if use_agent:
    st.markdown("### G. AI Decision Agent")
    mc_stats_ctx = mc_stats if 'mc_stats' in locals() else None
    regime_ctx = regime if 'regime' in locals() else None
    did_ctx = result if ('result' in locals() and 'result' is not None) else None

    ctx = build_agent_context(
        year=year,
        solver=("OSM A*" if routing_source.startswith("OSM") else solver),
        path=path,
        plen_km=float(dist_km),
        gps_spreader=gps_spreader,
        truck_co2_factor=float(truck_co2_factor),
        yield_df=yield_df,
        mc_stats=mc_stats_ctx,
        regime=regime_ctx,
        did_result=did_ctx,
        flow_df=flow_df,
        carbon_price=float(carbon_price),
        nitrates_per_ha=float(nitrates_per_ha),
        adoption_rate=int(adoption_rate)
    )

    cA, cB = st.columns([1,1])
    with cA:
        st.json(ctx, expanded=False)

    go = st.button("Run decision agent", key=wkey("run_agent"))
    if go:
        try:
            decision = call_openai_agent(model_choice, ctx)
            st.success("Agent decision received.")
            st.subheader("Decision")
            st.write(f"**Verdict:** {decision.get('route_verdict','?')}  |  **Objective:** {decision.get('primary_objective','balanced')}")
            st.markdown("**Rationale**")
            st.write(decision.get("rationale","(none)"))

            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Risk flags**")
                st.write(decision.get("risk_flags", []))
            with col2:
                st.markdown("**Recommended actions**")
                st.write(decision.get("recommended_actions", []))

            st.markdown("**KPIs**")
            st.json(decision.get("kpis", {}), expanded=False)

            st.markdown("**Sensitivity**")
            st.json(decision.get("sensitivity", {}), expanded=False)
        except Exception as e:
            st.error(f"Agent call failed: {e}")
            st.info("Check OPENAI_API_KEY, model availability, or reduce context size.")

st.markdown("---")
st.markdown("#### Notes")
st.markdown(
    "- **OSM A*** uses a directed drivable network with speeds/travel times; ports are treated as targets (no transit via other ports).\n"
    "- Synthetic solver prunes non-target ports for sink behavior; policy sliders affect synthetic edge costs via carbon externality & GPS rebate.\n"
    "- qLDPC here means ensemble+parity smoothing (intuitive stabilizer, not hardware QEC). Monte Carlo/Markov/DiD are scenario-aware."
)
