from typing import Dict

class Regulator:
    """
    Regulator class for adjusting sampling parameters based on uncertainty.
    Part of the Sensing-Regulating-Correcting loop (DenoiseFlow adaptation).
    """

    def __init__(self, base_temperature: float = 1.0, base_top_p: float = 1.0):
        self.base_temperature = base_temperature
        self.base_top_p = base_top_p

    def adjust_sampling(self, uncertainty_score: float) -> Dict[str, float]:
        """
        Adjust temperature and top_p based on an uncertainty score.
        High uncertainty (noise) leads to more deterministic sampling (lower values).
        Low uncertainty allows for more diverse sampling.

        Args:
            uncertainty_score: A float value representing uncertainty (0.0 to 1.0).

        Returns:
            Dictionary containing adjusted 'temperature' and 'top_p'.
        """
        # Clamp uncertainty score between 0 and 1
        score = max(0.0, min(1.0, uncertainty_score))

        # Heuristic: linear reduction of parameters as uncertainty increases
        # When score=1.0 (most uncertain), we want lowest temperature/top_p
        # When score=0.0 (least uncertain), we use base values
        
        # Min values for 'denoising'
        min_temp = 0.1
        min_top_p = 0.5

        temp = self.base_temperature - (score * (self.base_temperature - min_temp))
        top_p = self.base_top_p - (score * (self.base_top_p - min_top_p))

        return {
            "temperature": round(temp, 3),
            "top_p": round(top_p, 3)
        }
