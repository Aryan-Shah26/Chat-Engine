import time


class MetricsCollector:
    """Lightweight instrumentation collector for a single query lifecycle."""

    def __init__(self):
        self._timers: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self._values: dict[str, object] = {}

    def start_timer(self, label: str):
        self._starts[label] = time.perf_counter()

    def stop_timer(self, label: str):
        if label in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(label)
            self._timers[label] = round(elapsed, 4)

    def record(self, key: str, value):
        self._values[key] = value

    def to_dict(self) -> dict:
        """Return all collected metrics as a flat dict."""
        out = {}
        # Latency metrics
        for label, elapsed in self._timers.items():
            out[f"{label}_sec"] = elapsed
        # Recorded values
        out.update(self._values)
        # Derived: citation accuracy & hallucination rate
        extracted = self._values.get("citations_extracted", 0)
        verified = self._values.get("citations_verified", 0)
        if extracted > 0:
            accuracy = round(verified / extracted, 4)
            out["citation_accuracy"] = accuracy
            out["hallucination_rate"] = round(1 - accuracy, 4)
        else:
            out["citation_accuracy"] = None
            out["hallucination_rate"] = None
        return out
