"""
Display monthly_snapshots table data in a human-readable table.

Usage
-----
    # Show all rows (default 20)
    python scripts/show_monthly_snapshots.py

    # Show most recent N rows
    python scripts/show_monthly_snapshots.py --limit 5

    # Filter by cycle label
    python scripts/show_monthly_snapshots.py --cycle cycle-001

    # Show full detail for each row (one block per row)
    python scripts/show_monthly_snapshots.py --detail

Suitable for live demo — run from the repo root directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agritwin_gh.utils.database import get_db_manager
from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService


def _show_detail(session_factory, cycle: str | None, limit: int) -> None:
    """Print a detailed multi-line block for each snapshot row."""
    from sqlalchemy import text

    where = "WHERE cc.cycle_label = :cycle" if cycle else ""
    session = session_factory()
    try:
        rows = session.execute(text(f"""
            SELECT
                ms.*,
                cc.cycle_label,
                cc.started_at AS cycle_started_at,
                cc.completed  AS cycle_completed
            FROM monthly_snapshots ms
            JOIN crop_cycles cc ON cc.id = ms.cycle_id
            {where}
            ORDER BY ms.snapshot_recorded_at DESC
            LIMIT :lim
        """), {"lim": limit, "cycle": cycle} if cycle else {"lim": limit}).fetchall()

        if not rows:
            print("[show_monthly_snapshots] No rows found.")
            return

        cols = rows[0]._fields if hasattr(rows[0], "_fields") else []
        for r in rows:
            d = dict(zip(cols, r)) if cols else {}
            print("\n" + "═" * 60)
            print(f"  Cycle  : {d.get('cycle_label', '?')}   Month  : {d.get('billing_month', '?')}  ({d.get('month_label', '')})")
            print(f"  Period : {d.get('month_start_ts', '')} → {d.get('month_end_ts', '')}")
            print(f"  Steps  : {d.get('total_steps', 0):,}   MPC solves: {d.get('mpc_solve_count', 0):,}   Images: {d.get('image_classify_count', 0):,}")
            print(f"  Stage  : {d.get('stage_at_month_start', '?')} → {d.get('stage_at_month_end', '?')}  (transitions: {d.get('stage_transitions', 0)})")
            print("─" * 60)
            print(f"  Sensors (avg):  T={d.get('avg_indoor_temp', 0):.1f}°C  RH={d.get('avg_indoor_humidity', 0):.1f}%  CO₂={d.get('avg_co2', 0):.0f}ppm")
            print(f"                  Soil={d.get('avg_soil_moisture', 0):.1f}%  Light={d.get('avg_light_intensity', 0):.0f}lux")
            print(f"                  VPD={d.get('avg_vpd', 0):.2f}kPa  Risk={d.get('avg_disease_risk_score', 0):.3f}")
            print("─" * 60)
            print(f"  Resources:  Energy={d.get('total_energy_kwh', 0):.3f} kWh  "
                  f"Water={d.get('total_water_l', 0):.1f} L  "
                  f"Cost=₹{d.get('total_cost_inr', 0):.2f}")
            print(f"  Actuator breakdown (kWh):")
            print(f"    Fan={d.get('act_fan_speed_kwh', 0):.4f}  Vent={d.get('act_vent_opening_kwh', 0):.4f}  "
                  f"Heater={d.get('act_heater_output_kwh', 0):.4f}  LED={d.get('act_led_intensity_kwh', 0):.4f}")
            print(f"    Fogger={d.get('act_fogger_duty_kwh', 0):.4f}  CO₂v={d.get('act_co2_valve_kwh', 0):.4f}  "
                  f"Irrigation={d.get('act_irrigation_kwh', 0):.4f}kWh + {d.get('act_irrigation_water_l', 0):.1f}L")
            print("─" * 60)
            print(f"  Disease peaks:  EarlyBlight={d.get('peak_early_blight_sev', 0):.3f}  "
                  f"LateBlight={d.get('peak_late_blight_sev', 0):.3f}  "
                  f"LeafMold={d.get('peak_leaf_mold_sev', 0):.3f}")
            print(f"                  PowderyMildew={d.get('peak_powdery_mildew_sev', 0):.3f}  "
                  f"SpiderMites={d.get('peak_spider_mites_sev', 0):.3f}  "
                  f"AlertSteps={d.get('disease_alert_steps', 0)}")
            print("─" * 60)
            print(f"  Recorded at: {d.get('snapshot_recorded_at', 'unknown')}")
        print("═" * 60 + "\n")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display AgriTwin-GH monthly snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit", "-n", type=int, default=20, help="Max rows (default 20)")
    parser.add_argument("--cycle", "-c", type=str, default=None, help="Filter by cycle label, e.g. cycle-001")
    parser.add_argument("--detail", "-d", action="store_true", help="Show detailed multi-line output per row")
    args = parser.parse_args()

    mgr = get_db_manager()
    sf = mgr.get_session

    if args.detail:
        _show_detail(sf, cycle=args.cycle, limit=args.limit)
    else:
        MonthlySnapshotService.show_table(sf, limit=args.limit)


if __name__ == "__main__":
    main()
