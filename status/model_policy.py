from enum import Enum, auto
from typing import Optional, Dict

class ModelDecision(Enum):
    RUN_DOWNLOAD = auto()
    RUN_TRACE = auto()
    BYPASS = auto()
    ERROR = auto()


class ModelPolicy:
    """
    Model-level policy.

    Input:
      - status: dict | None  (from read_model_status)
      - force: bool

    Output:
      - ModelDecision
    """
    def __init__(self, *, force: bool = False):
        self.force = force

    def should_bypass_model(self, status: dict | None) -> bool:
        if self.force:
            return False

        if status is None:
            return False

        return status.get("stage") == "traced"

    def decide(self, status: dict | None) -> ModelDecision:
        """
        Decide what to do with a model.

        status["stage"] is expected to be one of:
          - downloaded
          - validated
          - traced
          - failed
        """

        if status is None:
            return ModelDecision.RUN_DOWNLOAD

        stage = status.get("stage")

        if stage == "traced":
            if self.force:
                return ModelDecision.RUN_TRACE
            return ModelDecision.BYPASS

        if stage == "validated":
            return ModelDecision.RUN_TRACE

        if stage in ("downloaded", "failed"):
            if self.force:
                return ModelDecision.RUN_DOWNLOAD
            return ModelDecision.ERROR

        return ModelDecision.ERROR
