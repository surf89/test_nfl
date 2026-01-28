"""
fantasy_engine.py

Backend engine for fantasy_app:
- Loads config
- Connects to ESPN
- Pulls NFL data with nfl_data_py
- Applies ESPN scoring (with bonuses)
- Engineers features (defense, volatility, dynamic boom/bust)
- Trains regression + boom + bust models
- Computes Start Score
- Provides:
    - weekly projections
    - start/sit
    - waiver targets
    - ROS projections
    - trade analysis
    - strength of schedule
    - league-winning move recommendations
    - matchup projections
    - ESPN-aware lineup optimizer with risk tolerance
    - weekly email report
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

import nfl_data_py as nfl
from espn_api.football import League


# =========================
# CONFIG + SEASON ROLLOVER
# =========================

def load_config(path: str = "fantasy_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def get_current_nfl_season():
    """
    NFL seasons roll over in September.
    Jan–Aug → previous season
    Sep–Dec → current year
    """
    today = datetime.today()
    year = today.year
    if today.month < 9:
        return year - 1
    return year


# =========================
# ESPN SCORING + LEAGUE
# =========================

def connect_espn_league(cfg):
    league = League(
        league_id=cfg["ESPN_LEAGUE_ID"],
        year=cfg["ESPN_YEAR"],
        espn_s2=cfg["ESPN_S2"],
        swid=cfg["ESPN_SWID"]
    )
    return league


def build_scoring_map(scoring):
    return {
        "passing_yards": scoring.get("passYds", 0),
        "passing_tds": scoring.get("passTds", 0),
        "interceptions": scoring.get("passInts", 0),
        "rushing_yards": scoring.get("rushYds", 0),
        "rushing_tds": scoring.get("rushTds", 0),
        "receiving_yards": scoring.get("recYds", 0),
        "receiving_tds": scoring.get("recTds", 0),
        "receptions": scoring.get("receptions", scoring.get("rec", 0)),
        "fumbles_lost": scoring.get("fumLost", 0),
        "pass_300_bonus": scoring.get("pass300", 0),
        "rush_100_bonus": scoring.get("rush100", 0),
        "rec_100_bonus": scoring.get("rec100", 0),
    }


def compute_espn_fantasy_points(row, scoring_map):
    pts = 0.0

    pts += row.get("passing_yards", 0) * scoring_map["passing_yards"]
    pts += row.get("passing_tds", 0) * scoring_map["passing_tds"]
    pts += row.get("interceptions", 0) * scoring_map["interceptions"]

    pts += row.get("rushing_yards", 0) * scoring_map["rushing_yards"]
    pts += row.get("rushing_tds", 0) * scoring_map["rushing_tds"]

    pts += row.get("receiving_yards", 0) * scoring_map["receiving_yards"]
    pts += row.get("receiving_tds", 0) * scoring_map["receiving_tds"]
    pts += row.get("receptions", 0) * scoring_map["receptions"]

    pts += row.get("fumbles_lost", 0) * scoring_map["fumbles_lost"]

    if row.get("passing_yards", 0) >= 300:
        pts += scoring_map["pass_300_bonus"]
    if row.get("rushing_yards", 0) >= 100:
        pts += scoring_map["rush_100_bonus"]
    if row.get("receiving_yards", 0) >= 100:
        pts += scoring_map["rec_100_bonus"]

    return pts


# =========================
# DATA PULL & PREP
# =========================

def pull_nfl_weekly_data(seasons, scoring_map):
    weekly = nfl.import_weekly_data(seasons)
    weekly = weekly[weekly["position"].isin(["QB", "RB", "WR", "TE"])]

    df = pd.DataFrame()
    df["season"] = weekly["season"]
    df["week"] = weekly["week"]
    df["player_id"] = weekly["player_id"]
    df["player_name"] = weekly["player_display_name"]
    df["position"] = weekly["position"]
    df["team"] = weekly["recent_team"]
    df["opponent_team"] = weekly["opponent_team"]

    df["passing_yards"] = weekly["passing_yards"]
    df["passing_tds"] = weekly["passing_tds"]
    df["interceptions"] = weekly["interceptions"]

    df["rushing_yards"] = weekly["rushing_yards"]
    df["rushing_tds"] = weekly["rushing_tds"]

    df["receiving_yards"] = weekly["receiving_yards"]
    df["receiving_tds"] = weekly["receiving_tds"]
    df["receptions"] = weekly["receptions"]

    df["fumbles_lost"] = weekly["fumbles_lost"]

    df["targets"] = weekly["targets"]
    df["carries"] = weekly["carries"]
    df["routes_run"] = weekly["routes_run"]
    df["snap_share"] = weekly["offense_snapshare"]
    df["red_zone_touches"] = weekly["redzone_targets"].fillna(0) + weekly["redzone_carries"].fillna(0)

    df["yards_per_route_run"] = weekly["yards_per_route_run"]
    df["yards_after_contact"] = weekly["yards_after_contact"]
    df["air_yards"] = weekly["air_yards"]

    df["fantasy_points"] = df.apply(
        lambda row: compute_espn_fantasy_points(row, scoring_map),
        axis=1
    )

    return df


def pull_team_defense_data(seasons):
    weekly = nfl.import_weekly_data(seasons)
    def_df = weekly.groupby(
        ["season", "week", "opponent_team"], as_index=False
    )["points"].sum()

    def_df = def_df.rename(columns={
        "opponent_team": "team",
        "points": "points_allowed"
    })

    return def_df


def add_opponent_defense_features(player_df, def_df):
    def_df_renamed = def_df.rename(columns={
        "team": "opponent_team",
        "points_allowed": "opp_points_allowed"
    })

    merged = player_df.merge(
        def_df_renamed,
        on=["season", "week", "opponent_team"],
        how="left"
    )

    merged = merged.sort_values(["opponent_team", "season", "week"])
    merged["opp_points_allowed_3g_avg"] = (
        merged.groupby("opponent_team")["opp_points_allowed"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return merged


def add_volatility_features(df):
    df = df.sort_values(["player_id", "season", "week"])
    df["usage"] = df["targets"].fillna(0) + df["carries"].fillna(0)

    df["fp_std_4"] = (
        df.groupby("player_id")["fantasy_points"]
        .rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["usage_std_4"] = (
        df.groupby("player_id")["usage"]
        .rolling(window=4, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["fp_std_4"] = df["fp_std_4"].fillna(0)
    df["usage_std_4"] = df["usage_std_4"].fillna(0)

    return df


def compute_position_thresholds(df, boom_pct=0.8, bust_pct=0.2):
    thresholds = []
    for pos, group in df.groupby("position"):
        boom_thr = np.percentile(group["fantasy_points"], boom_pct * 100)
        bust_thr = np.percentile(group["fantasy_points"], bust_pct * 100)
        thresholds.append({
            "position": pos,
            "boom_threshold": boom_thr,
            "bust_threshold": bust_thr,
        })
    return pd.DataFrame(thresholds)


def create_targets(df):
    thr_df = compute_position_thresholds(df)
    df = df.merge(thr_df, on="position", how="left")
    df["boom"] = (df["fantasy_points"] >= df["boom_threshold"]).astype(int)
    df["bust"] = (df["fantasy_points"] <= df["bust_threshold"]).astype(int)
    return df


def select_feature_columns(df):
    feature_cols = [
        "snap_share",
        "routes_run",
        "targets",
        "carries",
        "red_zone_touches",
        "yards_per_route_run",
        "yards_after_contact",
        "air_yards",
        "opp_points_allowed",
        "opp_points_allowed_3g_avg",
        "fp_std_4",
        "usage_std_4",
    ]
    return [c for c in feature_cols if c in df.columns]


def preprocess_for_model(df, feature_cols):
    df = df.dropna(subset=["fantasy_points"])

    cat_cols = [c for c in ["position", "team", "opponent_team"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    expanded_feature_cols = [c for c in feature_cols if c in df.columns]
    for col in df.columns:
        if any(col.startswith(base + "_") for base in ["position", "team", "opponent_team"]):
            expanded_feature_cols.append(col)

    X = df[expanded_feature_cols].fillna(0)
    y_points = df["fantasy_points"]
    y_boom = df["boom"]
    y_bust = df["bust"]

    return df, X, y_points, y_boom, y_bust, expanded_feature_cols


# =========================
# MODEL TRAINING
# =========================

def train_models(X, y_points, y_boom, y_bust, random_state=42):
    X_train, X_test, y_points_train, y_points_test = train_test_split(
        X, y_points, test_size=0.2, random_state=random_state
    )

    reg_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    reg_model.fit(X_train, y_points_train)
    y_pred_points = reg_model.predict(X_test)
    mae = mean_absolute_error(y_points_test, y_pred_points)
    print(f"[Regression] MAE (fantasy points): {mae:.2f}")

    boom_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    boom_clf.fit(X_train, y_boom.loc[X_train.index])
    y_proba_boom = boom_clf.predict_proba(X_test)[:, 1]
    try:
        boom_auc = roc_auc_score(y_boom.loc[X_test.index], y_proba_boom)
        print(f"[Classification] Boom AUC: {boom_auc:.3f}")
    except ValueError:
        print("Not enough class variation for boom AUC.")

    bust_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    bust_clf.fit(X_train, y_bust.loc[X_train.index])
    y_proba_bust = bust_clf.predict_proba(X_test)[:, 1]
    try:
        bust_auc = roc_auc_score(y_bust.loc[X_test.index], y_proba_bust)
        print(f"[Classification] Bust AUC: {bust_auc:.3f}")
    except ValueError:
        print("Not enough class variation for bust AUC.")

    return reg_model, boom_clf, bust_clf


# =========================
# ROSTER + WEEKLY DATA
# =========================

def build_upcoming_week_df(nfl_df, roster_df, season, week):
    latest = (
        nfl_df[nfl_df["season"] < season]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )

    merged = roster_df.merge(
        latest,
        on=["player_name", "team", "position"],
        how="left",
        suffixes=("", "_hist")
    )

    upcoming = merged.copy()
    upcoming["season"] = season
    upcoming["week"] = week

    cols_keep = [
        "season", "week", "player_id", "player_name", "position",
        "team", "opponent_team", "snap_share", "routes_run", "targets",
        "carries", "red_zone_touches", "yards_per_route_run",
        "yards_after_contact", "air_yards", "fp_std_4"
    ]
    cols_keep = [c for c in cols_keep if c in upcoming.columns]
    upcoming = upcoming[cols_keep]

    return upcoming


# =========================
# PREDICTION + START SCORE
# =========================

def prepare_upcoming_week_data(upcoming_df, def_df, feature_cols, trained_df_columns):
    upcoming_df = add_opponent_defense_features(upcoming_df, def_df)

    cat_cols = [c for c in ["position", "team", "opponent_team"] if c in upcoming_df.columns]
    upcoming_df = pd.get_dummies(upcoming_df, columns=cat_cols, drop_first=True)

    for col in trained_df_columns:
        if col not in upcoming_df.columns:
            upcoming_df[col] = 0

    upcoming_df = upcoming_df[trained_df_columns]
    X_upcoming = upcoming_df[feature_cols].fillna(0)
    return upcoming_df, X_upcoming


def compute_start_score(df):
    if "fp_std_4" in df.columns:
        vol = df["fp_std_4"].fillna(0)
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    else:
        vol_norm = 0

    df["start_score"] = (
        0.60 * df["projected_points"] +
        0.25 * (df["boom_probability"] * 100) -
        0.15 * (df["bust_probability"] * 100) -
        0.10 * (vol_norm * 100)
    )

    return df


def predict_week(reg_model, boom_clf, bust_clf,
                 upcoming_df, def_df, feature_cols, trained_df_columns):
    prepared_df, X_upcoming = prepare_upcoming_week_data(
        upcoming_df, def_df, feature_cols, trained_df_columns
    )

    projected_points = reg_model.predict(X_upcoming)
    boom_proba = boom_clf.predict_proba(X_upcoming)[:, 1]
    bust_proba = bust_clf.predict_proba(X_upcoming)[:, 1]

    prepared_df["projected_points"] = projected_points
    prepared_df["boom_probability"] = boom_proba
    prepared_df["bust_probability"] = bust_proba

    prepared_df = compute_start_score(prepared_df)

    return prepared_df


def build_start_sit_table(week_projections_df):
    df = week_projections_df.copy()
    df = df.sort_values(
        ["start_score", "projected_points", "boom_probability"],
        ascending=[False, False, False]
    )

    cols_to_show = [
        "player_name",
        "position",
        "team",
        "opponent_team",
        "projected_points",
        "boom_probability",
        "bust_probability",
        "start_score"
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    return df[cols_to_show]


# =========================
# WAIVER WIRE
# =========================

def find_waiver_targets(
    league,
    nfl_df,
    reg_model,
    boom_clf,
    bust_clf,
    def_df,
    feature_cols,
    trained_df_columns,
    target_season,
    target_week,
    top_n=10
):
    fa_list = league.free_agents(size=300)

    fa_rows = []
    for p in fa_list:
        fa_rows.append({
            "player_name": p.name,
            "position": p.position,
            "team": p.proTeam,
        })

    fa_df = pd.DataFrame(fa_rows)

    latest = (
        nfl_df[nfl_df["season"] < target_season]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )

    merged = fa_df.merge(
        latest,
        on=["player_name", "team", "position"],
        how="left"
    )

    merged["season"] = target_season
    merged["week"] = target_week

    upcoming_df = merged[
        [
            c for c in [
                "season", "week", "player_id", "player_name", "position",
                "team", "opponent_team", "snap_share", "routes_run",
                "targets", "carries", "red_zone_touches",
                "yards_per_route_run", "yards_after_contact",
                "air_yards", "fp_std_4"
            ] if c in merged.columns
        ]
    ]

    predictions = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        upcoming_df,
        def_df,
        feature_cols,
        trained_df_columns
    )

    trend_df = nfl_df.sort_values(["player_id", "season", "week"])
    trend_df["usage"] = trend_df["targets"].fillna(0) + trend_df["carries"].fillna(0)

    trend = (
        trend_df.groupby("player_id")["usage"]
        .apply(lambda x: x.tail(3).mean() - x.tail(6).head(3).mean()
               if len(x) >= 6 else 0)
        .reset_index()
        .rename(columns={"usage": "usage_trend"})
    )

    predictions = predictions.merge(trend, on="player_id", how="left")
    predictions["usage_trend"] = predictions["usage_trend"].fillna(0)

    predictions["waiver_score"] = (
        0.45 * predictions["start_score"] +
        0.35 * (predictions["boom_probability"] * 100) +
        0.20 * predictions["usage_trend"]
    )

    predictions = predictions.sort_values("waiver_score", ascending=False)

    return predictions.head(top_n)


# =========================
# ROS + TRADE ANALYZER
# =========================

def estimate_ros_for_players(
    reg_model,
    boom_clf,
    bust_clf,
    nfl_df,
    def_df,
    feature_cols,
    trained_df_columns,
    season,
    start_week,
    end_week=18
):
    latest = (
        nfl_df[nfl_df["season"] <= season]
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
    )

    ros_rows = []
    for wk in range(start_week, end_week + 1):
        week_df = latest.copy()
        week_df["season"] = season
        week_df["week"] = wk

        week_df = week_df[
            [
                c for c in [
                    "season", "week", "player_id", "player_name", "position",
                    "team", "opponent_team", "snap_share", "routes_run",
                    "targets", "carries", "red_zone_touches",
                    "yards_per_route_run", "yards_after_contact",
                    "air_yards", "fp_std_4"
                ] if c in week_df.columns
            ]
        ]

        preds = predict_week(
            reg_model,
            boom_clf,
            bust_clf,
            week_df,
            def_df,
            feature_cols,
            trained_df_columns
        )
        preds["proj_week"] = wk
        ros_rows.append(preds)

    ros_all = pd.concat(ros_rows, ignore_index=True)

    ros_summary = (
        ros_all.groupby(["player_id", "player_name", "position", "team"], as_index=False)
        .agg(
            ros_points=("projected_points", "mean"),
            ros_start_score=("start_score", "mean")
        )
    )

    return ros_summary


def get_team_ros(team, ros_df):
    names = [p.name for p in team.roster]
    return ros_df[ros_df["player_name"].isin(names)].copy()


def analyze_trade(
    my_team_ros,
    other_team_ros,
    my_out_names,
    my_in_names
):
    my_out = my_team_ros[my_team_ros["player_name"].isin(my_out_names)]
    my_in = other_team_ros[other_team_ros["player_name"].isin(my_in_names)]

    out_points = my_out["ros_points"].sum()
    in_points = my_in["ros_points"].sum()

    out_start = my_out["ros_start_score"].sum()
    in_start = my_in["ros_start_score"].sum()

    return {
        "out_points": out_points,
        "in_points": in_points,
        "delta_points": in_points - out_points,
        "out_start_score": out_start,
        "in_start_score": in_start,
        "delta_start_score": in_start - out_start,
    }


# =========================
# STRENGTH OF SCHEDULE
# =========================

def build_sos_table(nfl_df, season):
    df = nfl_df[nfl_df["season"] == season].copy()
    sos = (
        df.groupby(["opponent_team", "position"], as_index=False)
        .agg(fp_allowed=("fantasy_points", "mean"))
        .rename(columns={"opponent_team": "def_team"})
    )
    return sos


# =========================
# VISUAL HELPERS (logos/headshots)
# =========================

TEAM_LOGOS = {
    "BUF": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
}


def get_team_logo(team):
    return TEAM_LOGOS.get(team)


def get_player_headshot(player_name):
    safe_name = player_name.lower().replace(" ", "-")
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{safe_name}.png"


# =========================
# LEAGUE-WINNING MOVES
# =========================

def recommend_league_winning_moves(
    league,
    player_df,
    def_df,
    reg_model,
    boom_clf,
    bust_clf,
    feature_cols,
    trained_columns,
    cfg,
    top_n_waivers=5,
):
    waivers = find_waiver_targets(
        league,
        player_df,
        reg_model,
        boom_clf,
        bust_clf,
        def_df,
        feature_cols,
        trained_columns,
        cfg["TARGET_SEASON"],
        cfg["TARGET_WEEK"],
        top_n=top_n_waivers
    )

    ros_df = estimate_ros_for_players(
        reg_model,
        boom_clf,
        bust_clf,
        player_df,
        def_df,
        feature_cols,
        trained_columns,
        cfg["TARGET_SEASON"],
        cfg["TARGET_WEEK"]
    )

    my_team = league.teams[0]
    my_team_ros = get_team_ros(my_team, ros_df)

    my_weak_links = my_team_ros.sort_values("ros_start_score").head(top_n_waivers)
    waiver_best = waivers.sort_values("waiver_score", ascending=False).head(top_n_waivers)

    suggestions = []
    for (_, weak), (_, w) in zip(my_weak_links.iterrows(), waiver_best.iterrows()):
        suggestions.append({
            "drop": weak["player_name"],
            "add": w["player_name"],
            "delta_ros_points": w.get("projected_points", 0) - weak["ros_points"],
            "waiver_score": w["waiver_score"],
        })

    return {
        "waivers": waivers,
        "weak_links": my_weak_links,
        "suggested_moves": pd.DataFrame(suggestions),
    }


# =========================
# MATCHUP PROJECTION
# =========================

def get_current_week_matchup_projection(
    league,
    player_df,
    def_df,
    reg_model,
    boom_clf,
    bust_clf,
    feature_cols,
    trained_columns,
    season,
    week,
    my_team_index=0,
):
    box_scores = league.box_scores(week)
    my_team = league.teams[my_team_index]

    my_box = None
    opp_team = None
    for b in box_scores:
        if b.home_team.team_id == my_team.team_id:
            my_box = b
            opp_team = b.away_team
            break
        if b.away_team.team_id == my_team.team_id:
            my_box = b
            opp_team = b.home_team
            break

    if my_box is None or opp_team is None:
        return None

    def build_roster_df(team):
        rows = []
        for p in team.roster:
            rows.append(
                {
                    "player_name": p.name,
                    "position": p.position,
                    "team": p.proTeam,
                }
            )
        return pd.DataFrame(rows)

    my_roster_df = build_roster_df(my_team)
    opp_roster_df = build_roster_df(opp_team)

    my_upcoming = build_upcoming_week_df(player_df, my_roster_df, season, week)
    opp_upcoming = build_upcoming_week_df(player_df, opp_roster_df, season, week)

    my_proj = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        my_upcoming,
        def_df,
        feature_cols,
        trained_columns,
    )
    opp_proj = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        opp_upcoming,
        def_df,
        feature_cols,
        trained_columns,
    )

    my_total = my_proj["projected_points"].sum()
    opp_total = opp_proj["projected_points"].sum()

    return {
        "my_team_name": my_team.team_name,
        "opp_team_name": opp_team.team_name,
        "my_total": my_total,
        "opp_total": opp_total,
        "my_proj_df": my_proj,
        "opp_proj_df": opp_proj,
    }


# =========================
# RISK TOLERANCE + OPTIMIZER
# =========================

def apply_risk_tolerance(df, risk_level):
    """
    risk_level: 0 = safe, 0.5 = balanced, 1 = boom-chasing
    """
    df = df.copy()

    safe_adj = (
        -0.30 * (df["bust_probability"] * 100)
        -0.20 * df["fp_std_4"]
    )

    boom_adj = (
        +0.30 * (df["boom_probability"] * 100)
        +0.20 * df["projected_points"]
    )

    df["risk_adjusted_score"] = (
        df["start_score"]
        + (1 - risk_level) * safe_adj
        + (risk_level) * boom_adj
    )

    return df


def optimize_lineup_espm(
    league,
    player_df,
    def_df,
    reg_model,
    boom_clf,
    bust_clf,
    feature_cols,
    trained_columns,
    season,
    week,
    risk_level=0.5,
    my_team_index=0,
):
    my_team = league.teams[my_team_index]
    roster_settings = league.settings.roster_settings

    roster_rows = []
    for p in my_team.roster:
        roster_rows.append(
            {
                "player_name": p.name,
                "position": p.position,
                "team": p.proTeam,
                "eligible_slots": p.eligibleSlots,
            }
        )
    roster_df = pd.DataFrame(roster_rows)

    upcoming_df = build_upcoming_week_df(player_df, roster_df, season, week)
    proj = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        upcoming_df,
        def_df,
        feature_cols,
        trained_columns,
    )

    proj = apply_risk_tolerance(proj, risk_level)

    SLOT_MAP = {
        0: "QB",
        2: "RB",
        4: "WR",
        6: "TE",
        16: "D/ST",
        17: "K",
        20: "BENCH",
        21: "IR",
        23: "FLEX",       # RB/WR/TE
        24: "SUPERFLEX",  # QB/RB/WR/TE
    }

    slots = []
    for slot_id, count in roster_settings.items():
        if slot_id in SLOT_MAP and SLOT_MAP[slot_id] not in ["BENCH", "IR"]:
            slots.extend([SLOT_MAP[slot_id]] * count)

    used_ids = set()
    lineup = []

    def pick_best_for_slot(slot):
        eligible_positions = {
            "QB": ["QB"],
            "RB": ["RB"],
            "WR": ["WR"],
            "TE": ["TE"],
            "FLEX": ["RB", "WR", "TE"],
            "SUPERFLEX": ["QB", "RB", "WR", "TE"],
            "D/ST": ["D/ST"],
            "K": ["K"],
        }[slot]

        candidates = proj[
            (proj["position"].isin(eligible_positions))
            & (~proj["player_id"].isin(used_ids))
        ].sort_values("risk_adjusted_score", ascending=False)

        if candidates.empty:
            return None

        row = candidates.iloc[0]
        used_ids.add(row["player_id"])
        return row

    for slot in slots:
        row = pick_best_for_slot(slot)
        if row is not None:
            lineup.append({"slot": slot, **row.to_dict()})

    lineup_df = pd.DataFrame(lineup)
    total_proj = lineup_df["projected_points"].sum()
    bench_df = proj[~proj["player_id"].isin(used_ids)]

    return {
        "lineup_df": lineup_df,
        "bench_df": bench_df,
        "total_proj": total_proj,
        "full_proj_df": proj,
    }


# =========================
# WEEKLY EMAIL REPORT
# =========================

def build_weekly_email_body(
    league,
    player_df,
    def_df,
    reg_model,
    boom_clf,
    bust_clf,
    feature_cols,
    trained_columns,
    cfg,
    risk_level=0.5,
):
    season = cfg["TARGET_SEASON"]
    week = cfg["TARGET_WEEK"]

    opt = optimize_lineup_espm(
        league,
        player_df,
        def_df,
        reg_model,
        boom_clf,
        bust_clf,
        feature_cols,
        trained_columns,
        season,
        week,
        risk_level=risk_level,
    )

    matchup = get_current_week_matchup_projection(
        league,
        player_df,
        def_df,
        reg_model,
        boom_clf,
        bust_clf,
        feature_cols,
        trained_columns,
        season,
        week,
    )

    waivers = find_waiver_targets(
        league,
        player_df,
        reg_model,
        boom_clf,
        bust_clf,
        def_df,
        feature_cols,
        trained_columns,
        season,
        week,
        top_n=5,
    )

    lines = []
    lines.append(f"Weekly Fantasy Report – Season {season}, Week {week}")
    lines.append("")
    if matchup:
        lines.append(
            f"Matchup: {matchup['my_team_name']} vs {matchup['opp_team_name']}"
        )
        lines.append(
            f"Projected: {matchup['my_total']:.1f} – {matchup['opp_total']:.1f}"
        )
        lines.append("")

    lines.append("Optimized Lineup:")
    for _, row in opt["lineup_df"].iterrows():
        lines.append(
            f"  {row['slot']}: {row['player_name']} ({row['position']}, {row['team']}) "
            f"- {row['projected_points']:.1f} pts, StartScore {row['start_score']:.1f}"
        )
    lines.append(f"Total projected points: {opt['total_proj']:.1f}")
    lines.append("")

    lines.append("Top Waiver Targets:")
    for _, row in waivers.iterrows():
        lines.append(
            f"  {row['player_name']} ({row['position']}, {row['team']}) "
            f"- WaiverScore {row['waiver_score']:.1f}, Boom {row['boom_probability']:.2f}"
        )

    return "\n".join(lines)


def send_weekly_email_report(engine, risk_level=0.5):
    cfg = engine["config"]
    if not cfg.get("EMAIL_ENABLED", False):
        print("Email disabled in config.")
        return

    body = build_weekly_email_body(
        engine["league"],
        engine["player_df"],
        engine["def_df"],
        engine["reg_model"],
        engine["boom_clf"],
        engine["bust_clf"],
        engine["feature_cols"],
        engine["trained_columns"],
        cfg,
        risk_level=risk_level,
    )

    msg = MIMEMultipart()
    msg["From"] = cfg["EMAIL_FROM"]
    msg["To"] = cfg["EMAIL_TO"]
    msg["Subject"] = f"Fantasy Weekly Report – Week {cfg['TARGET_WEEK']}"

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(cfg["SMTP_SERVER"], cfg["SMTP_PORT"]) as server:
        server.starttls()
        server.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])
        server.send_message(msg)

    print("Weekly email report sent.")


# =========================
# ENGINE LOADER
# =========================

def load_engine():
    cfg = load_config()

    current_season = get_current_nfl_season()
    cfg["TARGET_SEASON"] = current_season
    cfg["ESPN_YEAR"] = current_season
    cfg["SEASONS"] = [current_season - 3, current_season - 2, current_season - 1]

    league = connect_espn_league(cfg)
    scoring_map = build_scoring_map(league.scoring_settings)

    player_df = pull_nfl_weekly_data(cfg["SEASONS"], scoring_map)
    def_df = pull_team_defense_data(cfg["SEASONS"])

    player_df = add_opponent_defense_features(player_df, def_df)
    player_df = add_volatility_features(player_df)
    player_df = create_targets(player_df)

    feature_cols = select_feature_columns(player_df)
    processed_df, X, y_points, y_boom, y_bust, expanded_feature_cols = preprocess_for_model(
        player_df, feature_cols
    )

    reg_model, boom_clf, bust_clf = train_models(
        X, y_points, y_boom, y_bust, random_state=cfg["RANDOM_STATE"]
    )

    return {
        "league": league,
        "player_df": player_df,
        "def_df": def_df,
        "reg_model": reg_model,
        "boom_clf": boom_clf,
        "bust_clf": bust_clf,
        "feature_cols": expanded_feature_cols,
        "trained_columns": processed_df.columns,
        "config": cfg
    }
