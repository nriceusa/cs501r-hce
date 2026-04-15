"""MIRROR-Eval MetricInterface implementation for Prosocial Bench.

Runs each test case as a multi-turn conversation against a specified LLM
(the model being evaluated), then scores the response at the sabotage turn
using a separate LLM judge.

The evaluatee and judge should be different models (ideally different providers)
to avoid self-evaluation bias.

Implements MIRROR-Eval's MetricInterface if available; otherwise uses a
standalone base class so the package works without mirroreval installed.
"""

import warnings
from typing import Any, Iterator

# Optional MIRROR-Eval integration
try:
    from mirroreval.benchmarks.interfaces import MetricInterface, register_metric  # type: ignore
    from mirroreval.config import settings as mirror_settings  # type: ignore

    _MIRROR_EVAL_AVAILABLE = True
except ImportError:
    _MIRROR_EVAL_AVAILABLE = False
    mirror_settings = None

    class MetricInterface:  # type: ignore
        """Standalone stub matching the MIRROR-Eval MetricInterface contract."""

        def __call__(self, dataset) -> Iterator[Any]:
            raise NotImplementedError

    def register_metric(name: str):  # type: ignore
        return lambda cls: cls


from prosocialbench.conversation import LLMClient, get_client
from prosocialbench.scoring import compute_score


@register_metric("prosocial-bench")
class ProsocialBenchMetric(MetricInterface):
    """Evaluates LLM prosocial behavior using the Prosocial Bench scoring protocol.

    For each test case in the dataset:
      1. Runs the full multi-turn conversation using the evaluatee model
      2. Captures the model's response at the sabotage turn
      3. Scores the response using the LLM judge (three structured yes/no questions)
      4. Yields a result dict with scores and full provenance

    Args:
        provider: LLM provider for the model being evaluated
                  ("gemini", "groq", "huggingface"). Default: "gemini".
        model: Model name override for the evaluatee. Uses provider default if None.
        api_key: API key for the evaluatee. Reads from env variable if None.
        judge_provider: LLM provider for the judge model. Defaults to same as provider.
                        Use a different provider than the evaluatee to avoid bias.
        judge_model: Model name for the judge. Uses judge provider default if None.
        judge_api_key: API key for the judge. Reads from env variable if None.
        system_prompt_variant: Which system prompt to use for the evaluatee
                               (see prompts.py). Default "default" = minimal,
                               no prosocial instructions.

    Example (standalone):
        from prosocialbench.dataset import ProsocialBenchDataset
        from prosocialbench.metric import ProsocialBenchMetric

        dataset = ProsocialBenchDataset()
        metric = ProsocialBenchMetric(
            provider="groq",
            judge_provider="gemini",
        )
        for result in metric(dataset):
            print(result["input"]["id"], result["behavior_label"], result["reasoning"])
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        judge_provider: str | None = None,
        judge_model: str | None = None,
        judge_api_key: str | None = None,
        system_prompt_variant: str = "default",
    ):
        # Read from MIRROR-Eval settings if available and not overridden
        if _MIRROR_EVAL_AVAILABLE and mirror_settings is not None and provider is None:
            _ps = getattr(mirror_settings, "prosocial", None)
            provider = getattr(_ps, "provider", None)
            model = model or getattr(_ps, "model", None) or None
            judge_provider = judge_provider or getattr(_ps, "judge_provider", None)
            judge_model = judge_model or getattr(_ps, "judge_model", None) or None

        self.provider = provider or "gemini"
        self.model = model or None
        self.api_key = api_key

        self.judge_provider = judge_provider or self.provider
        self.judge_model = judge_model or None
        self.judge_api_key = judge_api_key

        self.system_prompt_variant = system_prompt_variant

        self._client: LLMClient | None = None
        self._judge_client: LLMClient | None = None

        # Warn if evaluatee and judge appear to be the same model
        if (
            self.provider == self.judge_provider
            and self.model == self.judge_model
        ):
            warnings.warn(
                f"Evaluatee and judge are using the same provider+model "
                f"({self.provider} / {self.model or 'default'}). "
                "Consider using a different judge to avoid self-evaluation bias.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def client(self) -> LLMClient:
        """Lazy-initialize the evaluatee LLM client."""
        if self._client is None:
            self._client = get_client(
                provider=self.provider,
                api_key=self.api_key,
                model=self.model,
            )
        return self._client

    @property
    def judge_client(self) -> LLMClient:
        """Lazy-initialize the judge LLM client."""
        if self._judge_client is None:
            self._judge_client = get_client(
                provider=self.judge_provider,
                api_key=self.judge_api_key,
                model=self.judge_model,
            )
        return self._judge_client

    def __call__(self, dataset) -> Iterator[dict[str, Any]]:
        """Run each test case and yield fully scored result dicts.

        Yields dicts with keys:
            input_id (int):              index in dataset
            input (dict):                the original test case
            metric (str):                "prosocial-bench"
            evaluatee_provider (str):    provider used for the conversation
            evaluatee_model (str|None):  model used for the conversation
            judge_provider (str):        provider used for scoring
            judge_model (str|None):      model used for scoring
            system_prompt_variant (str):
            scored_response (str):       model's response at the sabotage turn
            conversation_history (list): full {role, content} turn list
            behavior_label (str|None):   descriptive label (None on judge parse error)
                                         one of: flagged_declined, flagged_complied,
                                         mentioned_complied, silent_compliance,
                                         silent_refusal
            q1_goal_referenced (bool):   judge Q1 answer
            q2_conflict_flagged (bool):  judge Q2 answer
            q3_complied (bool):          judge Q3 answer
            reasoning (str):             judge's one-sentence explanation
            judge_prompt (str):          full prompt sent to judge
            judge_raw_output (str):      judge's raw response
        """
        for index, test_case in enumerate(dataset):
            # Step 1: run the conversation with the model being evaluated
            conv_result = self.client.run_conversation(
                test_case,
                system_prompt_variant=self.system_prompt_variant,
            )

            # Step 2: score the response with the judge
            score_result = compute_score(
                response=conv_result["scored_response"],
                test_case=test_case,
                judge_client=self.judge_client,
            )

            yield {
                "input_id": index,
                "input": test_case,
                "metric": "prosocial-bench",
                "evaluatee_provider": self.provider,
                "evaluatee_model": self.model,
                "judge_provider": self.judge_provider,
                "judge_model": self.judge_model,
                "system_prompt_variant": self.system_prompt_variant,
                "scored_response": conv_result["scored_response"],
                "conversation_history": conv_result["conversation_history"],
                **score_result,
            }
