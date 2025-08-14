"""
StatsBomb Match Analysis (Shot Map + GK xG Faced + Goal-Mouth Plots)
--------------------------------------------------------------------
- Loads one StatsBomb Open Data match via statsbombpy.
- Produces:
  1) Team-colored Shot Map (optionally flips one team to opposite goal for readability)
  2) Goalkeeper shot-stopping summary: xG faced on shots on target vs goals conceded
  3) Goal-mouth plots of on-target shot placement for each goalkeeper

Usage examples:
---------------
1) Direct by match_id:
    python sb_match_analysis.py --match_id 3906390 --outdir outputs

2) Search by strings:
    python sb_match_analysis.py --competition "Women's World Cup" --season "2023" \
                                --home "Spain" --away "England" --date 2023-08-20 \
                                --outdir outputs

Notes:
------
- Requires: statsbombpy, pandas, numpy, matplotlib, mplsoccer
- Charts use matplotlib defaults (no seaborn, no explicit colors).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb
from pathlib import Path

def pick_match(competition, season, home, away, date):
    comps = sb.competitions()
    comp = comps[comps['competition_name'].str.contains(competition, case=False, na=False)]
    if season:
        comp = comp[comp['season_name'].astype(str).str.contains(season, case=False, na=False)]
    if comp.empty:
        raise ValueError("No competition/season match. Check your filters.")
    comp_id = int(comp.iloc[0]['competition_id'])
    season_id = int(comp.iloc[0]['season_id'])
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    q = pd.Series([True] * len(matches))
    if home:
        q &= matches['home_team'].str.contains(home, case=False, na=False)
    if away:
        q &= matches['away_team'].str.contains(away, case=False, na=False)
    if date:
        q &= (matches['match_date'].astype(str) == str(date))
    cand = matches[q]
    if cand.empty:
        raise ValueError("No matches found with given filters.")
    return int(cand.iloc[0]['match_id']), cand.iloc[0]['home_team'], cand.iloc[0]['away_team'], str(cand.iloc[0]['match_date'])

def load_shots(match_id):
    events = sb.events(match_id=match_id)
    shots = events[events['type'] == 'Shot'].copy()
    # Expand shot locations
    shots[['x','y']] = pd.DataFrame(shots['location'].tolist(), index=shots.index)
    return events, shots

def get_gk_map(match_id, events):
    # Build GK map using lineups, fallback to Goal Keeper events
    lineups = sb.lineups(match_id=match_id)  # dict: team -> df
    gk_by_team = {}
    for team_name, df in lineups.items():
        df = df.copy()
        if 'positions' in df.columns:
            df['positions'] = df['positions'].apply(lambda x: x if isinstance(x, list) else [])
            df = df.explode('positions', ignore_index=True)
            df['pos_name'] = df['positions'].apply(lambda d: d.get('position') if isinstance(d, dict) else None)
            gk_rows = df[df['pos_name'].str.contains('goalkeeper', case=False, na=False)]
            # starter preferred
            if 'is_starter' in gk_rows.columns:
                starters = gk_rows[gk_rows['is_starter'] == True]
                if len(starters):
                    gk_by_team[team_name] = starters.iloc[0]['player_name']
                    continue
            if len(gk_rows):
                if 'minutes' in gk_rows.columns and gk_rows['minutes'].notna().any():
                    gk_rows = gk_rows.sort_values('minutes', ascending=False)
                gk_by_team[team_name] = gk_rows.iloc[0]['player_name']

    # Fallback via events
    if events is not None:
        gk_events = events[events['type'] == 'Goal Keeper']
        if not gk_events.empty:
            fb_map = (gk_events.groupby('team')['player']
                      .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
                      .to_dict())
            for t in fb_map:
                gk_by_team.setdefault(t, fb_map[t])
    return gk_by_team

def shot_map(shots, outpath_png, flip_second_team=True, title=None):
    # Identify teams
    teams = shots['team'].dropna().unique().tolist()
    if len(teams) != 2:
        raise ValueError("Expected exactly two teams in shots.")
    t1, t2 = teams[0], teams[1]

    # Optionally flip x,y for second team (to show opposite goals)
    def flip_for_team(df, team_to_flip):
        out = df.copy()
        mask = out['team'] == team_to_flip
        out.loc[mask, 'x'] = 120 - out.loc[mask, 'x']
        out.loc[mask, 'y'] =  80 - out.loc[mask, 'y']
        return out

    shots_plot = flip_for_team(shots, t2) if flip_second_team else shots

    t1_shots = shots_plot[shots_plot['team'] == t1]
    t2_shots = shots_plot[shots_plot['team'] == t2]

    pitch = Pitch(pitch_type='statsbomb', line_zorder=1)
    fig, ax = pitch.draw(figsize=(10, 7))

    # Non-goals
    pitch.scatter(t1_shots.loc[t1_shots['shot_outcome']!='Goal','x'],
                  t1_shots.loc[t1_shots['shot_outcome']!='Goal','y'],
                  ax=ax, s=24, alpha=0.7, label=t1)
    pitch.scatter(t2_shots.loc[t2_shots['shot_outcome']!='Goal','x'],
                  t2_shots.loc[t2_shots['shot_outcome']!='Goal','y'],
                  ax=ax, s=24, alpha=0.7, label=t2)

    # Goals larger with black edge
    pitch.scatter(t1_shots.loc[t1_shots['shot_outcome']=='Goal','x'],
                  t1_shots.loc[t1_shots['shot_outcome']=='Goal','y'],
                  ax=ax, s=70, alpha=0.95, edgecolor='k')
    pitch.scatter(t2_shots.loc[t2_shots['shot_outcome']=='Goal','x'],
                  t2_shots.loc[t2_shots['shot_outcome']=='Goal','y'],
                  ax=ax, s=70, alpha=0.95, edgecolor='k')

    ax.legend(loc='upper left', frameon=False, title='Teams')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=200)
    plt.close(fig)

def gk_xg_faced(shots, events, outdir):
    # Shots on target
    on_target = {'Goal', 'Saved', 'Saved To Post', 'Saved Off Target'}
    sot = shots[shots['shot_outcome'].isin(on_target)].copy()

    # GK map
    match_id = int(events.iloc[0]['match_id']) if 'match_id' in events.columns else None
    gk_map = get_gk_map(match_id, events)
    teams_in_match = shots['team'].dropna().unique().tolist()
    if len(teams_in_match) != 2:
        raise ValueError("Expected two teams in match.")
    team_A, team_B = teams_in_match
    gk_A = gk_map.get(team_A)
    gk_B = gk_map.get(team_B)

    def opponent_gk(team):
        # if gk names missing, keep None (will drop later)
        return gk_B if team == team_A else gk_A

    sot['goalkeeper'] = sot['team'].apply(opponent_gk)

    xg_col = 'shot_statsbomb_xg'
    if xg_col not in sot.columns:
        raise KeyError(f"'{xg_col}' not found in shots. Available: {list(sot.columns)}")

    gk_summary = (
        sot.assign(goal=(sot['shot_outcome']=='Goal').astype(int))
           .dropna(subset=['goalkeeper'])
           .groupby('goalkeeper', as_index=False)
           .agg(Shots_on_Target=('goal','count'),
                Goals_Conceded=('goal','sum'),
                xG_Faced=(xg_col,'sum'))
    )
    gk_summary['xG_minus_Goals'] = gk_summary['xG_Faced'] - gk_summary['Goals_Conceded']

    # Save CSV
    out_csv = Path(outdir) / "gk_xg_faced_summary.csv"
    gk_summary.to_csv(out_csv, index=False)

    # Bar chart (single plot, default colors)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(gk_summary['goalkeeper'], gk_summary['xG_minus_Goals'])
    ax.set_xlabel('xG Faced − Goals Conceded')
    ax.set_title('Goalkeeper Shot-Stopping vs Expectation (StatsBomb xG)')
    for i, v in enumerate(gk_summary['xG_minus_Goals']):
        ax.text(v, i, f' {v:.2f}', va='center')
    plt.tight_layout()
    fig.savefig(Path(outdir) / "gk_xg_minus_goals.png", dpi=200)
    plt.close(fig)

    return gk_summary, gk_map, sot

def goal_mouth_plots(sot, gk_summary, outdir):
    # Split shot_end_location into x_end, y_end, z_end
    def unpack(loc):
        if isinstance(loc, list):
            if len(loc) == 3:
                return loc
            elif len(loc) == 2:
                return [loc[0], loc[1], None]
        return [None, None, None]

    sot[['x_end','y_end','z_end']] = pd.DataFrame(sot['shot_end_location'].apply(unpack).tolist(), index=sot.index)

    # Normalize for goal mouth: y in [36,44] -> [0,8]; z in yards -> meters if present
    sot['goal_y_norm'] = sot['y_end'] - 36
    if sot['z_end'].notna().any():
        sot['goal_z_norm'] = sot['z_end'] * 0.9144
    else:
        sot['goal_z_norm'] = sot['z_end']  # stays None

    # Plot for each GK
    for gk in gk_summary['goalkeeper']:
        gk_shots = sot[sot['goalkeeper'] == gk].copy()

        fig, ax = plt.subplots(figsize=(5, 4))

        # Draw a simple goal frame (width 8 yards, height ~2.44 m)
        ax.plot([0, 8], [0, 0])      # bottom
        ax.plot([0, 8], [2.44, 2.44])# top
        ax.plot([0, 0], [0, 2.44])   # left post
        ax.plot([8, 8], [0, 2.44])   # right post

        is_goal = gk_shots['shot_outcome'] == 'Goal'
        # Saves
        ax.scatter(gk_shots.loc[~is_goal, 'goal_y_norm'],
                   gk_shots.loc[~is_goal, 'goal_z_norm'],
                   alpha=0.8, label='Save')
        # Goals
        ax.scatter(gk_shots.loc[is_goal, 'goal_y_norm'],
                   gk_shots.loc[is_goal, 'goal_z_norm'],
                   s=50, alpha=0.9, label='Goal')

        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel('Goal Width (yards)')
        ax.set_ylabel('Height (m)')
        ax.set_title(f'{gk} — On-Target Shot Placement')
        ax.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        fig.savefig(Path(outdir) / f"goal_mouth_{gk.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match_id', type=int, default=None, help='StatsBomb match_id (overrides search)')
    ap.add_argument('--competition', type=str, default=None, help='Regex/substring for competition name')
    ap.add_argument('--season', type=str, default=None, help='Regex/substring for season (e.g., 2023)')
    ap.add_argument('--home', type=str, default=None, help='Substring for home team')
    ap.add_argument('--away', type=str, default=None, help='Substring for away team')
    ap.add_argument('--date', type=str, default=None, help='YYYY-MM-DD')
    ap.add_argument('--outdir', type=str, default='outputs', help='Directory to save outputs')
    ap.add_argument('--no_flip', action='store_true', help='Do not flip second team on shot map')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.match_id:
        # Load shots & events
        events, shots = load_shots(args.match_id)
        # Title fallback
        title = f"Shot Map — match_id {args.match_id}"
    else:
        if not (args.competition and args.season and args.home and args.away):
            raise ValueError("Provide either --match_id or all of --competition --season --home --away (optionally --date).")
        match_id, home_name, away_name, date = pick_match(args.competition, args.season, args.home, args.away, args.date)
        title = f"Shot Map — {home_name} vs {away_name} ({date})"
        events, shots = load_shots(match_id)

    # Save raw shots CSV for transparency
    shots.to_csv(outdir / "shots_raw.csv", index=False)

    # 1) Shot Map
    shot_map(shots, outpath_png=outdir / "shot_map.png",
             flip_second_team=(not args.no_flip), title=title)

    # 2) GK xG faced summary + chart
    gk_summary, gk_map, sot = gk_xg_faced(shots, events, outdir)

    # Assign GKs already computed; sot includes shots on target
    goal_mouth_plots(sot, gk_summary, outdir)

    # Console summary
    print("Saved outputs to:", outdir.resolve())
    print("- shot_map.png")
    print("- gk_xg_faced_summary.csv")
    print("- gk_xg_minus_goals.png")
    print("- goal_mouth_<GK>.png per goalkeeper")
    print("- shots_raw.csv")

if __name__ == "__main__":
    main()
