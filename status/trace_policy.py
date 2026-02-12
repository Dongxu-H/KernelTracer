import logging
from enum import Enum
from typing import Optional

class TraceDecision(Enum):
    RUN = "run"
    BYPASS = "bypass"


class TracePolicy:
    """
    Trace-level execution policy (PURE LOGIC).

    Inputs:
      - trace_status: dict | None   (from read_trace_status)
      - force: bool

    Handles:
      - if_exists (skip / overwrite / error)
      - bypass_if_exists (semantic completeness)
      - continue_on_failure
    """

    def __init__(self, behavior: dict, force: bool = False):
        self.force = force
        self.behavior = behavior or {}
        self.bypass_if_exists = behavior.get(
            "bypass_if_exists", True
        )
        self.continue_on_failure = behavior.get(
            "continue_on_failure", True
        )

    def decide(
        self,
        *,
        trace_status: Optional[dict],
        tracer_name: str,
        framework: str,
        device: str,
        mode: str,
    ) -> TraceDecision:
        """
        Decide what to do with this tracer.
        """

        # --------------------------------------------------------------
        # force = always re-run
        # --------------------------------------------------------------
        if self.force:
            logging.info("[trace-force] %s", tracer_name)
            return TraceDecision.RUN

        # --------------------------------------------------------------
        # never traced
        # --------------------------------------------------------------
        if trace_status is None:
            return TraceDecision.RUN

        # --------------------------------------------------------------
        # previous failure → retry
        # --------------------------------------------------------------
        if not trace_status.get("success", False):
            logging.info("[trace-retry] %s", tracer_name)
            return TraceDecision.RUN

        # --------------------------------------------------------------
        # success exists
        # --------------------------------------------------------------
        if not self.bypass_if_exists:
            return TraceDecision.RUN

        # semantic match check
        if (
            trace_status.get("tracer") == tracer_name
            and trace_status.get("framework") == framework
            and trace_status.get("device") == device
            and trace_status.get("mode") == mode
        ):
            logging.info("[trace-bypass] %s", tracer_name)
            return TraceDecision.BYPASS

        return TraceDecision.RUN
    
    # ------------------------------------------------------------------
    # failure policy
    # ------------------------------------------------------------------

    def should_continue_on_failure(self) -> bool:
        return bool(self.continue_on_failure)
