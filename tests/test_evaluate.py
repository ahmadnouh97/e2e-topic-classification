import unittest

import numpy as np
import pandas as pd

from topic_classification import evaluate


class SliceMetricsTests(unittest.TestCase):
    def test_slice_metrics_preserve_membership_and_scores(self):
        emoji = next(iter(evaluate.data.emoji_dict.values()))
        df = pd.DataFrame({"text": [emoji, "one two three four five six seven eight nine ten"]})
        metrics = evaluate.get_metrics(
            np.array([0, 1]), np.array([0, 0]), ["a", "b"], df
        )
        self.assertEqual(metrics["slices"]["with_emojis"]["num_samples"], 1)
        self.assertEqual(metrics["slices"]["short_tweet"]["num_samples"], 1)
        self.assertEqual(metrics["slices"]["short_tweet"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
