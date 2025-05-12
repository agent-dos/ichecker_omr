# app/features/analyzer/processors/bubble_processor.py (new file)
"""
Bubble processing utilities.
"""
import numpy as np
from typing import List, Tuple, Dict


class BubbleProcessor:
    """Processes and groups bubbles."""

    def __init__(self, params: Dict):
        self.params = params
        self.row_threshold = params.get('row_threshold', 8)

    def group_bubbles(
        self,
        bubbles: np.ndarray,
        image_width: int
    ) -> List[Tuple[int, List[np.ndarray]]]:
        """
        Group bubbles by question rows.
        """
        midpoint_x = image_width // 2
        left_bubbles = [b for b in bubbles if b[0] < midpoint_x]
        right_bubbles = [b for b in bubbles if b[0] >= midpoint_x]

        # Process columns
        items_per_column = 30
        columns = [
            (left_bubbles, 1),
            (right_bubbles, items_per_column + 1)
        ]

        grouped = []
        for column_bubbles, start_q in columns:
            if not column_bubbles:
                continue

            rows = self._group_into_rows(column_bubbles)
            for row_idx, row_bubbles in enumerate(rows):
                q_num = start_q + row_idx
                grouped.append((q_num, row_bubbles))

        return grouped

    def _group_into_rows(
        self,
        bubbles: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Group bubbles into rows based on Y position.
        """
        y_sorted = sorted(bubbles, key=lambda b: b[1])

        rows = []
        current_row = [y_sorted[0]]

        for i in range(1, len(y_sorted)):
            if abs(y_sorted[i][1] - current_row[0][1]) < self.row_threshold:
                current_row.append(y_sorted[i])
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [y_sorted[i]]

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        rows.sort(key=lambda r: r[0][1] if r else 0)
        return rows
