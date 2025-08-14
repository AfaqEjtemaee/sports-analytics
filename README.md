# StatsBomb Match Analysis — Shot Map + GK xG Faced + Goal-Mouth

This script loads a single **StatsBomb Open Data** match and produces:
1) **Shot Map** (teams in different colors; optional flipping so each team shoots at opposite goals)
2) **Goalkeeper shot-stopping summary**: xG faced on shots on target vs goals conceded
3) **Goal-mouth plots** of shot placement faced by each goalkeeper

> Charts use **matplotlib** only. No seaborn. One chart per plot. No explicit colors.

## Usage

### Option A : Direct by match_id
```bash
python sb_match_analysis.py --match_id 3906390 --outdir outputs
```

### Option B : Search by strings
```bash
python sb_match_analysis.py \
  --competition "Women's World Cup" --season "2023" \
  --home "Spain" --away "England" --date 2023-08-20 \
  --outdir outputs
```

## Outputs
- `shot_map.png` : team-colored shot map (StatsBomb pitch)
- `gk_xg_faced_summary.csv` : keeper summary table
- `gk_xg_minus_goals.png` : bar chart of xG faced minus goals conceded
- `goal_mouth_<GK>.png` : goal-mouth plots for each GK’s on-target shots faced
- `shots_raw.csv` : raw shot events for transparency

## Method Notes
- **Shot Map:** StatsBomb normalizes attacks to the right goal; the script optionally flips the second team for readability.
- **GK Metric:** Public StatsBomb data lacks PSxG, so we use **xG faced** on shots on target vs **goals conceded**. Positive `(xG − Goals)` suggests above-expected shot-stopping.
- **Goal-mouth:** Uses `shot_end_location` to plot placement in an 8-yard-wide, ~2.44 m high goal frame.

---


