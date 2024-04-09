"""Sampling parameters for text generation."""
from typing import List, Optional, Union
import json

_SAMPLING_EPS = 1e-5


class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        max_output_len
        temperature
        top_p
        top_k
        beam_width
        repetition_penalty
        presence_penalty
        len_penalty
        min_length
        random_seed
        end_id
    """

    def __init__(
        self,
            max_output_len: int = 16,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            beam_width: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            len_penalty: Optional[float] = None,
            random_seed: Optional[int] = None,
            end_id: Optional[List[int]] = None,
            pad_id: Optional[List[int]] = None,
            stop_words: Optional[List[str]] = None,
            **kwargs: object
    ) -> None:
        self.max_output_len = max_output_len
        self.beam_width = beam_width
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.len_penalty = len_penalty
        self.random_seed = random_seed
        self.end_id = end_id
        self.pad_id = pad_id
        self.stop_words = stop_words

        self._verify_args()
        # if self.beam_width:
        #     self._verity_beam_search()
        # elif self.temperature is not None and self.temperature < _SAMPLING_EPS:
        #     # Zero temperature means greedy sampling.
        #     self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if self.repetition_penalty:
            if not -2.0 <= self.repetition_penalty <= 2.0:
                raise ValueError("repetition_penalty must be in [-2, 2], got "
                                 f"{self.repetition_penalty}.")
        if self.presence_penalty:
            if not -2.0 <= self.presence_penalty <= 2.0:
                raise ValueError("presence_penalty must be in [-2, 2], got "
                                 f"{self.presence_penalty}.")
        if self.temperature is not None and self.temperature < 0.0:
                raise ValueError(
                    f"temperature must be non-negative, got {self.temperature}.")
        if self.top_p is not None:
            if not 0.0 < self.top_p <= 1.0:
                raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k is not None:
            if self.top_k < -1 or self.top_k == 0:
                raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                                 f"got {self.top_k}.")
        if self.max_output_len < 1:
            raise ValueError(
                f"max_output_len must be at least 1, got {self.max_output_len}.")

    def _verity_beam_search(self) -> None:
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.top_p is not None and self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_p is not None and self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")


    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, )
