# app/features/analyzer/steps/base.py
"""Base class for analysis pipeline steps."""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class AnalysisStep(ABC):
    """Abstract base class for analysis steps."""

    def __init__(self, params: Dict):
        """Initialize step with parameters."""
        self.params = params
        self.step_name = self.__class__.__name__

    @abstractmethod
    def process(self, context: Dict) -> Dict:
        """
        Process the step with given context.

        Args:
            context: Dictionary containing processing state

        Returns:
            Dictionary with step_info and context_update
        """
        pass

    def create_step_info(self, description: str, success: bool,
                         input_image: np.ndarray, output_image: np.ndarray,
                         viz_steps: Dict = None) -> Dict:
        """Create standard step info structure."""
        return {
            'name': self.step_name,
            'description': description,
            'success': success,
            'input_image': input_image,
            'output_image': output_image,
            'intermediate_visualizations': viz_steps or {}
        }
