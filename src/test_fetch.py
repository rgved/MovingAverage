"""
test_fetch.py — Tests for the smart incremental fetch logic in fetch-data-upstox.py

Tests covered:
  1. BOOTSTRAP  — No existing file → requests BOOTSTRAP_DAYS (~3 months)
  2. UP-TO-DATE — File exists, last date is today → skips entirely
  3. INCREMENTAL — File exists, last date is a few days ago → requests only missing days
  4. STALE      — File exists but last date > STALE_THRESHOLD_DAYS ago → re-fetches full 3 months
  5. MERGE      — Incremental fetch correctly merges + deduplicates rows

Run with:  python src/test_fetch.py
"""

import os
import sys
import csv
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta, date
from unittest.mock import patch, MagicMock

# ── Make sure the src package is importable ──────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

# fetch-data-upstox.py has hyphens in the filename, so it cannot be imported
# with a standard `import` statement.  We use importlib.util to load it by
# file path and expose it as the module alias `fdu`.
os.environ.setdefault("UPSTOX_ACCESS_TOKEN", "TEST_TOKEN_DUMMY")

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "fetch_data_upstox",
    os.path.join(SRC_DIR, "fetch-data-upstox.py"),
)
fdu = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(fdu)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_csv(path: str, last_date: date, num_rows: int = 10):
    """Write a minimal OHLCV CSV with `num_rows` rows ending on `last_date`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for i in range(num_rows):
            d = last_date - timedelta(days=(num_rows - 1 - i))
            writer.writerow([
                datetime.combine(d, datetime.min.time()).isoformat(),
                100, 105, 95, 102, 1000000
            ])
            
    # Backdate the file's modification time to yesterday so it bypasses 
    # the new "checked today" fast-path in fetch_history.
    yesterday_ts = (datetime.now() - timedelta(days=1)).timestamp()
    os.utime(path, (yesterday_ts, yesterday_ts))


def _fake_candles(from_date: date, to_date: date) -> list:
    """Generate synthetic candle data for the requested date range."""
    candles = []
    current = from_date
    while current < to_date:
        ts = datetime.combine(current, datetime.min.time()).isoformat() + "+05:30"
        candles.append([ts, 100.0, 110.0, 90.0, 105.0, 500000, 0])
        current += timedelta(days=1)
    return candles


# ── Test Cases ────────────────────────────────────────────────────────────────

class TestFetchModeSelection(unittest.TestCase):

    def setUp(self):
        """Create a temporary DATA_DIR for each test."""
        self.tmp_dir = tempfile.mkdtemp()
        # Redirect the module's DATA_DIR to our temp directory
        self._orig_data_dir = fdu.DATA_DIR
        fdu.DATA_DIR = self.tmp_dir

    def tearDown(self):
        fdu.DATA_DIR = self._orig_data_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── 1. BOOTSTRAP ──────────────────────────────────────────────────────────
    def test_bootstrap_when_no_file_exists(self):
        """If no raw CSV exists, from_date should be ~BOOTSTRAP_DAYS ago."""
        today = datetime.today().date()
        expected_from = today - timedelta(days=fdu.BOOTSTRAP_DAYS)

        captured_from = {}

        def fake_fetch(instrument_key, from_date, to_date):
            captured_from["from_date"] = from_date
            return _fake_candles(from_date, to_date)

        with patch.object(fdu, "_fetch_candles", side_effect=fake_fetch):
            fdu.fetch_history("TEST.NS", "NSE_EQ|TEST")

        self.assertEqual(captured_from["from_date"], expected_from,
                         "Bootstrap should request exactly BOOTSTRAP_DAYS back.")

        # Verify file was created
        out_path = os.path.join(self.tmp_dir, "TEST.NS.csv")
        self.assertTrue(os.path.exists(out_path), "CSV should have been written.")

    # ── 2. UP-TO-DATE ─────────────────────────────────────────────────────────
    def test_skip_when_already_up_to_date(self):
        """If the last row's date is today, fetch should be skipped entirely."""
        today = datetime.today().date()
        path = os.path.join(self.tmp_dir, "TEST.NS.csv")
        _make_csv(path, last_date=today)

        fetch_called = {"called": False}

        def fake_fetch(*args, **kwargs):
            fetch_called["called"] = True
            return []

        with patch.object(fdu, "_fetch_candles", side_effect=fake_fetch):
            fdu.fetch_history("TEST.NS", "NSE_EQ|TEST")

        self.assertFalse(fetch_called["called"],
                         "API should NOT be called when data is already up-to-date.")

    # ── 3. INCREMENTAL ────────────────────────────────────────────────────────
    def test_incremental_fetch_for_recent_file(self):
        """File exists with data 3 days old → should request only recent days."""
        today = datetime.today().date()
        last_stored = today - timedelta(days=3)

        path = os.path.join(self.tmp_dir, "TEST.NS.csv")
        _make_csv(path, last_date=last_stored)

        captured_from = {}

        def fake_fetch(instrument_key, from_date, to_date):
            captured_from["from_date"] = from_date
            return _fake_candles(from_date, to_date)

        with patch.object(fdu, "_fetch_candles", side_effect=fake_fetch):
            fdu.fetch_history("TEST.NS", "NSE_EQ|TEST")

        # from_date should be last_stored - INCREMENTAL_BUFFER_DAYS (NOT 3 months ago)
        expected_from = last_stored - timedelta(days=fdu.INCREMENTAL_BUFFER_DAYS)
        self.assertEqual(captured_from["from_date"], expected_from,
                         "Incremental fetch should only request a small recent window.")

        # from_date should be much more recent than bootstrap would give
        bootstrap_from = today - timedelta(days=fdu.BOOTSTRAP_DAYS)
        self.assertGreater(captured_from["from_date"], bootstrap_from,
                           "Incremental from_date must be much newer than a bootstrap from_date.")

    # ── 4. STALE ──────────────────────────────────────────────────────────────
    def test_stale_file_triggers_full_refetch(self):
        """File is older than STALE_THRESHOLD_DAYS → should re-fetch BOOTSTRAP_DAYS."""
        today = datetime.today().date()
        last_stored = today - timedelta(days=fdu.STALE_THRESHOLD_DAYS + 5)

        path = os.path.join(self.tmp_dir, "TEST.NS.csv")
        _make_csv(path, last_date=last_stored)

        captured_from = {}

        def fake_fetch(instrument_key, from_date, to_date):
            captured_from["from_date"] = from_date
            return _fake_candles(from_date, to_date)

        with patch.object(fdu, "_fetch_candles", side_effect=fake_fetch):
            fdu.fetch_history("TEST.NS", "NSE_EQ|TEST")

        expected_from = today - timedelta(days=fdu.BOOTSTRAP_DAYS)
        self.assertEqual(captured_from["from_date"], expected_from,
                         "Stale file should trigger a full BOOTSTRAP_DAYS re-fetch.")

    # ── 5. MERGE & DEDUPLICATION ──────────────────────────────────────────────
    def test_incremental_merge_deduplicates_rows(self):
        """Overlapping rows from the buffer window should not be double-counted."""
        today = datetime.today().date()
        last_stored = today - timedelta(days=3)

        path = os.path.join(self.tmp_dir, "TEST.NS.csv")
        _make_csv(path, last_date=last_stored, num_rows=10)

        import pandas as pd
        existing_rows = pd.read_csv(path)
        initial_count = len(existing_rows)

        def fake_fetch(instrument_key, from_date, to_date):
            # Deliberately overlap with existing data (buffer causes this)
            return _fake_candles(from_date, to_date)

        with patch.object(fdu, "_fetch_candles", side_effect=fake_fetch):
            fdu.fetch_history("TEST.NS", "NSE_EQ|TEST")

        result_df = pd.read_csv(path)
        result_df["Date"] = pd.to_datetime(result_df["Date"])

        # No duplicate dates
        self.assertEqual(
            len(result_df["Date"].unique()),
            len(result_df),
            "Merged CSV must not contain duplicate dates."
        )

        # Total rows should be >= initial (new rows added, not duplicated)
        self.assertGreaterEqual(len(result_df), initial_count,
                                "Merged CSV should have at least as many rows as before.")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  fetch-data-upstox.py — Smart Fetch Logic Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
