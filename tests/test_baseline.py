import json
import unittest
from pathlib import Path

from run_baseline import answer_question, build_vectors


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = json.loads(
            (PROJECT_ROOT / "examples/course_faq.json").read_text()
        )
        cls.vectors, cls.idf = build_vectors(cls.records)

    def test_concrete_examples_match_expected_sources(self) -> None:
        cases = {
            "When do I need to submit the capstone proposal?": "capstone_due_date",
            "What files should my README mention?": "readme_requirements",
            "Can the baseline be simple?": "baseline_requirement",
        }

        for question, expected_source in cases.items():
            with self.subTest(question=question):
                result = answer_question(
                    question, self.records, self.vectors, self.idf
                )
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["matched_source"], expected_source)

    def test_unrelated_question_requests_clarification(self) -> None:
        result = answer_question(
            "How do I bake sourdough bread?",
            self.records,
            self.vectors,
            self.idf,
        )

        self.assertEqual(result["status"], "needs_clarification")
        self.assertIsNone(result["matched_source"])


if __name__ == "__main__":
    unittest.main()
