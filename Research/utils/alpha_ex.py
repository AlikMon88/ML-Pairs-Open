
class PairsTradingAlpha:
    """
    Generates raw, {-1, 0, 1} trading signals for pairs based on Z-scores.
    This module is pure alpha logic and knows nothing about risk or size.
    """
    def __init__(self, pairs_data: dict, lookback_window: int, entry_z: float, exit_z: float):
        self.pairs_data = pairs_data
        self.lookback = lookback_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_scores = self._precompute_z_scores()

    def _precompute_z_scores(self) -> dict:
        """Pre-calculates Z-scores for efficiency."""
        z_scores = {}
        for pair, data in self.pairs_data.items():
            spread = data['spread_series']
            mean = spread.rolling(window=self.lookback).mean()
            std = spread.rolling(window=self.lookback).std()
            z_scores[pair] = (spread - mean) / std
        return z_scores

    def generate_signal(self, timestamp: pd.Timestamp, current_positions: dict) -> dict:
        """
        For a given timestamp, determine the ideal target position.

        Args:
            timestamp: The current time point of the backtest.
            current_positions: A dict of current positions, e.g., {'pair_1': 1}.

        Returns:
            dict: The desired target position, e.g., {'pair_1': 0}.
        """
        target_positions = {}
        for pair, z_series in self.z_scores.items():
            prev_pos = current_positions.get(pair, 0)
            target_pos = prev_pos
            
            if timestamp in z_series.index:
                z = z_series.loc[timestamp]
                if pd.isna(z):
                    target_pos = prev_pos # Not enough data, hold position
                elif prev_pos == 0:
                    if z < -self.entry_z: target_pos = 1  # Long the pair
                    elif z > self.entry_z: target_pos = -1 # Short the pair
                elif prev_pos == 1 and z > -self.exit_z:
                    target_pos = 0  # Exit long
                elif prev_pos == -1 and z < self.exit_z:
                    target_pos = 0  # Exit short
            
            target_positions[pair] = target_pos
            
        return target_positions