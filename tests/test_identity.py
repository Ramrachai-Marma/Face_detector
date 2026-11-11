import unittest

from advanced.identity import find_label_id


class FindLabelIdTests(unittest.TestCase):
    def test_matches_name_without_case_or_whitespace_differences(self):
        self.assertEqual(find_label_id({"0": "Myself"}, " myself "), 0)

    def test_returns_none_when_requested_name_is_not_present(self):
        self.assertIsNone(find_label_id({"0": "Myself"}, "Mark"))

    def test_does_not_assume_a_single_other_identity_is_the_requested_person(self):
        self.assertIsNone(find_label_id({"0": "Myself"}, "Unknown"))


if __name__ == "__main__":
    unittest.main()
