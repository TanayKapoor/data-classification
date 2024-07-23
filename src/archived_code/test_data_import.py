import unittest

class TestDataImport(unittest.TestCase):
    def test_generate_dq_score(self):
        dq_report = {
            "missing_values": {"col1": 10, "col2": 5},
            "unique_values": [1, 2, 3],
            "duplicate_rows": 20,
            "rows": 100
        }

        expected_dq_score = 100 - (10 / 100) * 100 - (5 / 100) * 100 - 10 - (20 / 100) * 100

        actual_dq_score = generate_dq_score(dq_report)

        self.assertEqual(actual_dq_score, expected_dq_score)

if __name__ == '__main__':
    unittest.main()