from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from daydream.utils import format_time_ago


class UtilsTests(unittest.TestCase):
    def test_format_time_ago_accepts_datetime(self) -> None:
        value = datetime.now(timezone.utc) - timedelta(minutes=5)
        self.assertEqual(format_time_ago(value), "5 min ago")

    def test_format_time_ago_accepts_timestamp(self) -> None:
        value = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        self.assertEqual(format_time_ago(value), "2 hours ago")


if __name__ == "__main__":
    unittest.main()
