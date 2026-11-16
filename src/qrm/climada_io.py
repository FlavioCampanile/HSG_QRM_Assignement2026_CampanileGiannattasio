"""CLIMADA adapter: from a winter-storm hazard to the storm catalogue used by the overlay  [Q xi].

This is the only module that imports CLIMADA.  It follows the hazard-exposure-vulnerability logic
of the ETH "mandatory exercise 1" referred to in the assignment:

    hazard         probabilistic set of European winter storms restricted to Switzerland,
                   client.get_hazard("storm_europe", properties={"country_iso3alpha": "CHE"});
                   event e has an annual frequency f_e and a footprint of maximum gust speeds (m/s);
    exposure       the 100 counterparties at their coordinates (and, as a robustness check, at the
                   centroid of their canton);
    vulnerability  the impact function of Welker et al. (2021), ImpfStormEurope.from_welker(), which
                   maps gust speed into the mean damage ratio MDR = MDD x PAA.

MDR_k(e) is the impact function evaluated at the gust speed of event e at the hazard centroid
nearest to counterparty k.  This is exactly what CLIMADA's ImpactCalc computes for an exposure of
value 1 at that point; `cross_check_impactcalc` verifies it.
"""
import datetime as _dt
import numpy as np
import pandas as pd
from .climate import StormCatalogue

HAZARD_TYPE = "storm_europe"
HAZARD_PROPERTIES = {"country_iso3alpha": "CHE"}


# ------------------------------------------------------------------ hazard
def load_hazard(path=None):
    """Return (hazard, info).  Without `path` the hazard is downloaded through the CLIMADA data API
    (it is cached by CLIMADA in ~/climada/data, so the download happens only once)."""
    if path is not None:
        from climada.hazard import Hazard
        return Hazard.from_hdf5(str(path)), {"dataset": f"file {path}"}
    from climada.util.api_client import Client
    client = Client()
    info = {"dataset": f"CLIMADA data API: {HAZARD_TYPE}, {HAZARD_PROPERTIES}"}
    try:
        haz = client.get_hazard(HAZARD_TYPE, properties=HAZARD_PROPERTIES)
        try:
            ds = client.get_dataset_info(data_type=HAZARD_TYPE, properties=HAZARD_PROPERTIES)
            info.update(dataset_name=str(ds.name), dataset_version=str(ds.version),
                        dataset_properties={k: str(v) for k, v in dict(ds.properties).items()})
        except Exception as err:                                  # metadata only: never fatal
            info["dataset_info_error"] = repr(err)
    except Exception as err:
        # Typically: several data sets match the properties.  List them and take the first by name,
        # so that the choice is deterministic and visible in the log.
        infos = sorted(client.list_dataset_infos(data_type=HAZARD_TYPE, properties=HAZARD_PROPERTIES),
                       key=lambda d: (str(d.name), str(d.version)))
        if not infos:
            raise
        print(f"get_hazard failed with {err!r}; {len(infos)} matching data sets:")
        for d in infos:
            print("   ", d.name, d.version, dict(d.properties))
        ds = infos[0]
        print("-> using", ds.name, ds.version)
        haz = client.get_hazard(HAZARD_TYPE, name=ds.name, version=ds.version, properties=HAZARD_PROPERTIES)
        info.update(dataset_name=str(ds.name), dataset_version=str(ds.version),
                    dataset_properties={k: str(v) for k, v in dict(ds.properties).items()},
                    n_matching_datasets=len(infos))
    return haz, info


def centroid_coords(hazard):
    """Latitude and longitude of the hazard centroids (attribute names differ across CLIMADA versions)."""
    c = hazard.centroids
    try:
        return np.asarray(c.lat, float), np.asarray(c.lon, float)
    except Exception:
        coord = np.asarray(c.coord, float)
        return coord[:, 0], coord[:, 1]


def nearest_centroid(clat, clon, lat, lon):
    """Index of the nearest hazard centroid for every point, and the great-circle distance in km."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    idx, dist = np.empty(len(lat), dtype=int), np.empty(len(lat))
    for i, (la, lo) in enumerate(zip(lat, lon)):
        h = (np.sin(np.deg2rad(clat - la) / 2) ** 2
             + np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(clat)) * np.sin(np.deg2rad(clon - lo) / 2) ** 2)
        d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))    # haversine
        idx[i], dist[i] = int(np.argmin(d)), float(d.min())
    return idx, dist


# ------------------------------------------------------------------ vulnerability
def welker_impact_function():
    from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
    return ImpfStormEurope.from_welker()


def mdr_of(impf, intensity):
    """Mean damage ratio MDR = MDD x PAA at the given intensities (linear interpolation on the
    support points of the impact function, as in ImpactFunc.calc_mdr)."""
    x = np.asarray(impf.intensity, float)
    y = np.asarray(impf.mdd, float) * np.asarray(impf.paa, float)
    return np.interp(np.asarray(intensity, float), x, y)


# ------------------------------------------------------------------ catalogue
def build_catalogue(hazard, info, df, cantons, impf=None):
    """Storm catalogue for the portfolio `df` (columns latitude, longitude, canton_abbr, E, R).

    `cantons` is the 'Swiss Cantons' sheet (canton_abbr, centroid_lat, centroid_lon)."""
    impf = welker_impact_function() if impf is None else impf
    clat, clon = centroid_coords(hazard)
    freq = np.asarray(hazard.frequency, float)
    n_events = len(freq)

    # counterparty level: gust speed at the nearest hazard centroid
    idx, dist = nearest_centroid(clat, clon, df["latitude"], df["longitude"])
    inten = np.asarray(hazard.intensity[:, idx].todense(), float)            # (n_events, K)
    mdr = mdr_of(impf, inten)

    # canton level: gust speed at the canton centroid, shared by all counterparties of the canton
    cc = cantons.set_index("canton_abbr").loc[df["canton_abbr"]]
    idx_c, dist_c = nearest_centroid(clat, clon, cc["centroid_lat"], cc["centroid_lon"])
    inten_c = np.asarray(hazard.intensity[:, idx_c].todense(), float)
    mdr_c = mdr_of(impf, inten_c)

    # the single most damaging event for this portfolio (largest loss of recovery value), for the map
    recov = (df["E"] * df["R"]).to_numpy()
    top = int(np.argmax(mdr @ recov))
    row = lambda m: np.asarray(m.todense(), float).ravel()
    names = np.array([str(x) for x in getattr(hazard, "event_name", range(n_events))])
    dates = np.array([_dt.date.fromordinal(int(d)).isoformat() if int(d) > 0 else ""
                      for d in np.asarray(getattr(hazard, "date", np.zeros(n_events)), dtype=np.int64)])
    orig = np.asarray(getattr(hazard, "orig", np.zeros(n_events, dtype=bool)), dtype=bool)
    extra = {"inten": inten.astype(np.float32), "inten_canton": inten_c.astype(np.float32),
             "impf_intensity": np.asarray(impf.intensity, float),
             "impf_mdr": np.asarray(impf.mdd, float) * np.asarray(impf.paa, float),
             "impf_mdd": np.asarray(impf.mdd, float), "impf_paa": np.asarray(impf.paa, float),
             "cent_lat": clat.astype(np.float32), "cent_lon": clon.astype(np.float32),
             "footprint_top": row(hazard.intensity[top]).astype(np.float32),
             "footprint_max": row(hazard.intensity.max(axis=0)).astype(np.float32),
             "event_name": names, "event_date": dates, "event_orig": orig,
             "nearest_km": dist, "nearest_km_canton": dist_c, "top_event": np.array(top)}
    try:
        from importlib.metadata import version
        climada_version = version("climada")
    except Exception:
        climada_version = "unknown"
    meta = dict(info)
    meta.update({"synthetic": False, "climada_version": climada_version,
                 "impact_function": f"{type(impf).__name__}: {getattr(impf, 'name', '')}".strip(),
                 "hazard_type": str(getattr(hazard, "haz_type", "")),
                 "intensity_unit": str(getattr(hazard, "units", "")),
                 "frequency_unit": str(getattr(hazard, "frequency_unit", "1/year")),
                 "n_events": int(n_events), "n_centroids": int(len(clat)), "n_orig_events": int(orig.sum()),
                 "top_event_name": str(names[top]), "top_event_date": str(dates[top])})
    return StormCatalogue(freq=freq, mdr=mdr, mdr_canton=mdr_c, extra=extra, meta=meta)


def cross_check_impactcalc(hazard, df, mdr, impf=None):
    """Re-compute MDR_k(e) with CLIMADA's own impact engine: an exposure of value 1 at every
    counterparty, impact matrix = MDR.  Returns the largest absolute difference to `mdr`."""
    from climada.entity import Exposures, ImpactFuncSet
    from climada.engine import ImpactCalc
    impf = welker_impact_function() if impf is None else impf
    col = f"impf_{hazard.haz_type}"
    gdf = pd.DataFrame({"latitude": df["latitude"].to_numpy(), "longitude": df["longitude"].to_numpy(),
                        "value": 1.0, col: int(impf.id)})
    try:
        exp = Exposures(gdf)
        if hasattr(exp, "set_geometry_points"):
            exp.set_geometry_points()
    except Exception:                                             # newer constructor signature
        exp = Exposures(data={"value": gdf["value"].to_numpy(), col: gdf[col].to_numpy()},
                        lat=gdf["latitude"].to_numpy(), lon=gdf["longitude"].to_numpy())
    exp.check()
    impact = ImpactCalc(exp, ImpactFuncSet([impf]), hazard).impact(save_mat=True)
    mdr_cc = np.asarray(impact.imp_mat.todense(), float)
    return float(np.abs(mdr_cc - mdr).max()), float(impact.aai_agg)
