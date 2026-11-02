import os
import tempfile
import unittest

from pilots import jobs

LOG = """[2026-10-30T09:58:54Z] job=x GPU snapshot (5 samples):
[2026-10-30T09:58:54Z] job=x chose GPU 1 (auto: least-loaded)
[2026-10-30T09:58:54Z] job=x session=s started
some output without timestamp
[2026-10-30T10:04:26Z] job=x finished exit=0
"""


class TestJobs(unittest.TestCase):
    def test_rows_and_total(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.log")
            with open(p, "w") as f:
                f.write(LOG)
            r = jobs.rows(["CPU:x:" + p, "CUDA:y:" + p])
        self.assertEqual(r[1].split("\t"), ["x", "2026-10-30T09:58:54Z", "2026-10-30T10:04:26Z", "332", "0.092", "0", "1", "CPU"])
        self.assertEqual(r[-1], "TOTAL_R4\t\t\t664\t0.184")


if __name__ == "__main__":
    unittest.main()
