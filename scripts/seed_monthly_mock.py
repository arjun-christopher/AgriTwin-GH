"""
Seed the monthly_snapshots and crop_cycles tables with demo data.

Usage
-----
    python scripts/seed_monthly_mock.py

Inserts 2 crop cycles (cycle-001 and cycle-002) with 3 monthly snapshot
rows containing realistic greenhouse telemetry values.  Safe to run
multiple times — uses ON CONFLICT DO NOTHING.

Prerequisites
-------------
1. Apply the schema first:
       psql -d <dbname> -f database/schema/monthly_snapshots.sql
   or for SQLite, ensure the tables exist.

2. Set database settings in ``config/settings.local.yaml``
   (or let the default SQLite path be used).
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agritwin_gh.utils.database import get_db_manager
from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService


def main() -> None:
    mgr = get_db_manager()

    MonthlySnapshotService.seed_mock_data(mgr.get_session)

    print("\nTo view the seeded data run:")
    print("    python scripts/show_monthly_snapshots.py\n")


if __name__ == "__main__":
    main()
