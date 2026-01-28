import streamlit as st
import pandas as pd
import altair as alt

from fantasy_engine import (
    load_engine,
    build_upcoming_week_df,
    predict_week,
    build_start_sit_table,
    find_waiver_targets,
    estimate_ros_for_players,
    get_team_ros,
    analyze_trade,
    build_sos_table,
    recommend_league_winning_moves,
    get_team_logo,
    get_player_headshot,
    get_current_week_matchup_projection,
    optimize_lineup_espm,
    send_weekly_email_report,
)

st.set_page_config(page_title="Fantasy Football Dashboard", layout="wide")


@st.cache_resource
def load_all():
    return load_engine()


engine = load_all()

league = engine["league"]
player_df = engine["player_df"]
def_df = engine["def_df"]
reg_model = engine["reg_model"]
boom_clf = engine["boom_clf"]
bust_clf = engine["bust_clf"]
feature_cols = engine["feature_cols"]
trained_columns = engine["trained_columns"]
CFG = engine["config"]


def add_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "team" in df.columns:
        df["team_logo"] = df["team"].apply(get_team_logo)
    if "player_name" in df.columns:
        df["headshot"] = df["player_name"].apply(get_player_headshot)
    return df


def style_start_sit(df: pd.DataFrame):
    styled = df.style.background_gradient(
        subset=["projected_points", "start_score"],
        cmap="Greens"
    ).background_gradient(
        subset=["boom_probability"],
        cmap="Blues"
    ).background_gradient(
        subset=["bust_probability"],
        cmap="Reds"
    )
    return styled


st.title("🏈 Fantasy Football Analytics Dashboard")

# Sidebar controls
st.sidebar.header("Season Controls")

season_options = list(range(2021, CFG["TARGET_SEASON"] + 2))
selected_season = st.sidebar.selectbox(
    "Select NFL Season",
    options=season_options,
    index=season_options.index(CFG["TARGET_SEASON"])
)

selected_week = st.sidebar.selectbox(
    "Select Week",
    options=list(range(1, 19)),
    index=CFG["TARGET_WEEK"] - 1
)

risk_level = st.sidebar.slider(
    "Risk Tolerance",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = Safe, 1 = Boom-chasing"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Weekly Projections",
        "Start/Sit",
        "Waiver Wire",
        "Trade Analyzer",
        "Rest-of-Season (ROS)",
        "Strength of Schedule",
        "League-Winning Moves",
        "Matchup & Lineup",
    ]
)

# -------------------------
# Weekly Projections
# -------------------------
if page == "Weekly Projections":
    st.header("📅 Weekly Projections")

    my_team = league.teams[0]
    roster_df = pd.DataFrame(
        [{"player_name": p.name, "position": p.position, "team": p.proTeam} for p in my_team.roster]
    )

    upcoming_df = build_upcoming_week_df(
        player_df,
        roster_df,
        selected_season,
        selected_week
    )

    projections = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        upcoming_df,
        def_df,
        feature_cols,
        trained_columns
    )

    projections = add_visual_columns(projections)

    st.dataframe(projections, use_container_width=True)

    chart_data = projections[["player_name", "projected_points", "boom_probability", "bust_probability"]]
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=120)
        .encode(
            x="projected_points",
            y="boom_probability",
            color="bust_probability",
            tooltip=["player_name", "projected_points", "boom_probability", "bust_probability"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Start/Sit
# -------------------------
elif page == "Start/Sit":
    st.header("🟢 Start/Sit Recommendations")

    my_team = league.teams[0]
    roster_df = pd.DataFrame(
        [{"player_name": p.name, "position": p.position, "team": p.proTeam} for p in my_team.roster]
    )

    upcoming_df = build_upcoming_week_df(
        player_df,
        roster_df,
        selected_season,
        selected_week
    )

    projections = predict_week(
        reg_model,
        boom_clf,
        bust_clf,
        upcoming_df,
        def_df,
        feature_cols,
        trained_columns
    )

    start_sit = build_start_sit_table(projections)
    start_sit = add_visual_columns(start_sit)

    st.dataframe(style_start_sit(start_sit), use_container_width=True)

    chart_data = start_sit[["player_name", "projected_points", "boom_probability", "bust_probability"]]
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=120)
        .encode(
            x="projected_points",
            y="boom_probability",
            color="bust_probability",
            tooltip=["player_name", "projected_points", "boom_probability", "bust_probability"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Waiver Wire
# -------------------------
elif page == "Waiver Wire":
    st.header("📈 Waiver Wire Targets")

    waivers = find_waiver_targets(
        league,
        player_df,
        reg_model,
        boom_clf,
        bust_clf,
        def_df,
        feature_cols,
        trained_columns,
        selected_season,
        selected_week,
        top_n=15
    )

    waivers = add_visual_columns(waivers)

    st.dataframe(
        waivers[
            [
                "player_name",
                "position",
                "team",
                "usage_trend",
                "boom_probability",
                "start_score",
                "waiver_score",
            ]
        ],
        use_container_width=True,
    )

    chart_data = waivers[["player_name", "usage_trend", "waiver_score"]]
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=120)
        .encode(
            x="usage_trend",
            y="waiver_score",
            tooltip=["player_name", "usage_trend", "waiver_score"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Trade Analyzer
# -------------------------
elif page == "Trade Analyzer":
    st.header("🔄 Trade Analyzer")

    ros_df = estimate_ros_for_players(
        reg_model,
        boom_clf,
        bust_clf,
        player_df,
        def_df,
        feature_cols,
        trained_columns,
        selected_season,
        selected_week
    )

    my_team = league.teams[0]
    other_teams = league.teams[1:]

    my_team_ros = get_team_ros(my_team, ros_df)

    st.subheader("Your Players (ROS)")
    st.dataframe(my_team_ros, use_container_width=True)

    team_names = [t.team_name for t in other_teams]
    selected_team = st.selectbox("Select a team to trade with", team_names)

    other_team = next(t for t in other_teams if t.team_name == selected_team)
    other_team_ros = get_team_ros(other_team, ros_df)

    st.subheader(f"{selected_team}'s Players (ROS)")
    st.dataframe(other_team_ros, use_container_width=True)

    my_out = st.multiselect("Players you give", my_team_ros["player_name"])
    my_in = st.multiselect("Players you receive", other_team_ros["player_name"])

    if st.button("Analyze Trade"):
        if not my_out or not my_in:
            st.warning("Select at least one player to give and receive.")
        else:
            result = analyze_trade(my_team_ros, other_team_ros, my_out, my_in)
            st.write("You give:", my_out)
            st.write("You receive:", my_in)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROS Points Δ", f"{result['delta_points']:.2f}")
            with col2:
                st.metric("ROS Start Score Δ", f"{result['delta_start_score']:.2f}")

# -------------------------
# ROS
# -------------------------
elif page == "Rest-of-Season (ROS)":
    st.header("📆 Rest-of-Season Projections")

    ros_df = estimate_ros_for_players(
        reg_model,
        boom_clf,
        bust_clf,
        player_df,
        def_df,
        feature_cols,
        trained_columns,
        selected_season,
        selected_week
    )

    ros_df = ros_df.sort_values("ros_points", ascending=False)
    st.dataframe(ros_df, use_container_width=True)

    chart_data = ros_df[["player_name", "ros_points", "ros_start_score"]]
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=120)
        .encode(
            x="ros_points",
            y="ros_start_score",
            tooltip=["player_name", "ros_points", "ros_start_score"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Strength of Schedule
# -------------------------
elif page == "Strength of Schedule":
    st.header("🛡️ Strength of Schedule (SoS)")

    sos_df = build_sos_table(player_df, selected_season)
    sos_df = sos_df.sort_values("fp_allowed", ascending=False)

    st.dataframe(sos_df, use_container_width=True)

    chart = (
        alt.Chart(sos_df)
        .mark_bar()
        .encode(
            x="def_team",
            y="fp_allowed",
            color="position",
            tooltip=["def_team", "position", "fp_allowed"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# League-Winning Moves
# -------------------------
elif page == "League-Winning Moves":
    st.header("🏆 League-Winning Moves")

    rec = recommend_league_winning_moves(
        league,
        player_df,
        def_df,
        reg_model,
        boom_clf,
        bust_clf,
        feature_cols,
        trained_columns,
        CFG,
    )

    st.subheader("Top Waiver Targets")
    st.dataframe(rec["waivers"], use_container_width=True)

    st.subheader("Weak Links on Your Roster (ROS)")
    st.dataframe(rec["weak_links"], use_container_width=True)

    st.subheader("Suggested Add/Drop Moves")
    st.dataframe(rec["suggested_moves"], use_container_width=True)

# -------------------------
# Matchup & Lineup
# -------------------------
elif page == "Matchup & Lineup":
    st.header("📊 Matchup Projection & Lineup Optimizer")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matchup Projection")
        matchup = get_current_week_matchup_projection(
            league,
            player_df,
            def_df,
            reg_model,
            boom_clf,
            bust_clf,
            feature_cols,
            trained_columns,
            selected_season,
            selected_week,
        )
        if matchup is None:
            st.write("No matchup found for this week.")
        else:
            st.metric(
                f"{matchup['my_team_name']} projected",
                f"{matchup['my_total']:.1f}",
            )
            st.metric(
                f"{matchup['opp_team_name']} projected",
                f"{matchup['opp_total']:.1f}",
            )
            st.write("Your players:")
            st.dataframe(matchup["my_proj_df"], use_container_width=True)
            st.write("Opponent players:")
            st.dataframe(matchup["opp_proj_df"], use_container_width=True)

    with col2:
        st.subheader("Optimized Lineup")
        opt = optimize_lineup_espm(
            league,
            player_df,
            def_df,
            reg_model,
            boom_clf,
            bust_clf,
            feature_cols,
            trained_columns,
            selected_season,
            selected_week,
            risk_level=risk_level,
        )
        st.dataframe(opt["lineup_df"], use_container_width=True)
        st.metric("Total projected points", f"{opt['total_proj']:.1f}")

        st.subheader("Bench")
        st.dataframe(opt["bench_df"], use_container_width=True)

    st.subheader("Weekly Email Report")
    if st.button("Send Weekly Email (manual trigger)"):
        send_weekly_email_report(engine, risk_level=risk_level)
        st.success("Email triggered (check logs for any SMTP issues).")
