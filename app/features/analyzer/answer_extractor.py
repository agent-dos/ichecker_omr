# app/features/analyzer/answer_extractor.py
"""
Extracts final answers from bubble analysis results.
"""
from typing import List, Tuple, Dict


class AnswerExtractor:
    """
    Extracts answers from analyzed bubble scores.
    """

    def extract(self, bubble_scores: List[Dict]) -> List[Tuple[int, str]]:
        """
        Extract final answers from bubble scores.

        Args:
            bubble_scores: List of analyzed bubble data

        Returns:
            List of (question_number, answer) tuples
        """
        answers = []

        for score_data in bubble_scores:
            q_num = score_data['question_number']
            choices = score_data['choices']

            # Find selected answer
            answer = ""
            for choice in choices:
                if choice.get('selected', False):
                    answer = choice['label']
                    break

            answers.append((q_num, answer))

        # Sort by question number
        answers.sort(key=lambda x: x[0])

        return answers
