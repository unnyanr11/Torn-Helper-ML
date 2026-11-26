import unittest
from unittest.mock import Mock
import json
from urllib.parse import urlparse, parse_qs

from python.stock_ml.historical_backfill import (
    TICKERS, fetch_chunk, minute_chunks, parse_rows,
)


class BackfillTests(unittest.TestCase):
    def test_all_35_stocks_are_assigned_to_18_shards(self):
        self.assertEqual(len(TICKERS), 35)
        for project in range(1, 18):
            self.assertEqual([project * 2 - 1, project * 2],
                             [project * 2 - 1, project * 2])
        self.assertEqual(TICKERS[35], "PTS")

    def test_chunks_are_minute_aligned_non_overlapping_and_bounded(self):
        chunks = list(minute_chunks(120, 120 + 4000 * 60, 1800))
        self.assertEqual(chunks, [(120, 108120), (108120, 216120), (216120, 240120)])
        self.assertTrue(all(a % 60 == 0 and b % 60 == 0 and b > a for a, b in chunks))

    def test_parse_rows_filters_invalid_and_deduplicates_real_minutes(self):
        payload = {"data": [
            [120, "10.5", 1000], [180, "11.0", 1100],
            [180, "11.2", 1200], [240, "-1", 1300],
            [300, "12", 1400], [301, "13", 1500],
            ["bad", "13", 1500], [360, None, 1500],
        ]}
        rows = parse_rows(payload, 120, 300)
        self.assertEqual(rows, [(120, 10.5, 1000), (180, 11.2, 1200)])

    def test_fetch_chunk_uses_tornsy_minute_endpoint_and_bounds(self):
        body = json.dumps({"data": [[120, "10.5", 1000], [180, "11.0", 1100]]}).encode()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read = Mock(return_value=body)
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return response

        rows = fetch_chunk("TSB", 120, 240, opener=opener, limiter=lambda: None)
        query = parse_qs(urlparse(captured["url"]).query)
        self.assertEqual(query["interval"], ["m1"])
        self.assertEqual(query["from"], ["120"])
        self.assertEqual(query["to"], ["240"])
        self.assertEqual(query["limit"], ["2000"])
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
