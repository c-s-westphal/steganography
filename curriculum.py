"""
Curriculum learning for the colors experiment.

Implements progressive difficulty from the paper:
"beginning with two colors and two names, then progressing to three colors
and three names, and so on until all colors are successfully mapped."
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from prompts import ALL_COLORS, ALL_NAMES


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Starting number of colors/names
    start_n: int = 2

    # Maximum number of colors/names
    max_n: int = 8

    # Accuracy threshold to advance to next level
    advance_threshold: float = 0.8

    # Number of episodes to evaluate before deciding to advance
    eval_window: int = 50

    # Minimum episodes at each level before advancing
    min_episodes_per_level: int = 100


class CurriculumManager:
    """
    Manages curriculum progression for the colors experiment.

    Tracks performance and advances difficulty when threshold is met.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_n = config.start_n
        self.episode_count = 0
        self.level_episode_count = 0
        self.recent_results: List[bool] = []  # Recent success/failure
        self.level_history: List[dict] = []

    def get_current_colors(self) -> List[str]:
        """Get the current set of colors for this curriculum level."""
        return ALL_COLORS[:self.current_n]

    def get_current_names(self) -> List[str]:
        """Get the current set of names for this curriculum level."""
        return ALL_NAMES[:self.current_n]

    def record_result(self, success: bool):
        """Record the result of an episode."""
        self.episode_count += 1
        self.level_episode_count += 1
        self.recent_results.append(success)

        # Keep only recent results for evaluation
        if len(self.recent_results) > self.config.eval_window:
            self.recent_results = self.recent_results[-self.config.eval_window:]

    def should_advance(self) -> bool:
        """Check if we should advance to the next curriculum level."""
        # Don't advance if at max
        if self.current_n >= self.config.max_n:
            return False

        # Need minimum episodes at current level
        if self.level_episode_count < self.config.min_episodes_per_level:
            return False

        # Need enough recent results
        if len(self.recent_results) < self.config.eval_window:
            return False

        # Check accuracy threshold
        accuracy = sum(self.recent_results) / len(self.recent_results)
        return accuracy >= self.config.advance_threshold

    def advance(self):
        """Advance to the next curriculum level."""
        accuracy = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.0

        self.level_history.append({
            "level": self.current_n,
            "episodes": self.level_episode_count,
            "final_accuracy": accuracy
        })

        self.current_n += 1
        self.level_episode_count = 0
        self.recent_results = []

        print(f"\n{'='*60}")
        print(f"CURRICULUM ADVANCE: Now using {self.current_n} colors/names")
        print(f"{'='*60}")

    def get_current_accuracy(self) -> float:
        """Get the current rolling accuracy."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def get_status(self) -> dict:
        """Get current curriculum status."""
        return {
            "current_n": self.current_n,
            "total_episodes": self.episode_count,
            "level_episodes": self.level_episode_count,
            "current_accuracy": self.get_current_accuracy(),
            "colors": self.get_current_colors(),
            "names": self.get_current_names()
        }

    def is_complete(self) -> bool:
        """Check if curriculum is complete (reached max level with good accuracy)."""
        if self.current_n < self.config.max_n:
            return False

        if len(self.recent_results) < self.config.eval_window:
            return False

        accuracy = sum(self.recent_results) / len(self.recent_results)
        return accuracy >= self.config.advance_threshold


def sample_color_task(
    colors: List[str],
    names: List[str]
) -> Tuple[str, List[str], List[str]]:
    """
    Sample a random color for encoding.

    Returns:
        Tuple of (color_to_encode, shuffled_colors, shuffled_names)
    """
    color = np.random.choice(colors)

    # Shuffle lists to prevent positional encoding
    shuffled_colors = colors.copy()
    shuffled_names = names.copy()
    np.random.shuffle(shuffled_colors)
    np.random.shuffle(shuffled_names)

    return color, shuffled_colors, shuffled_names
