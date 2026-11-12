"""Metered forward-pass accounting. One forward_pass call (a batch of prompts) costs one unit."""
import threading


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, total: int, max_prompts_per_pass: int):
        self.total, self.used = int(total), 0
        self.max_prompts = int(max_prompts_per_pass)
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return self.total - self.used

    def charge(self, n_prompts: int):
        with self._lock:
            if n_prompts < 1:
                raise ValueError("forward_pass needs at least one prompt")
            if n_prompts > self.max_prompts:
                raise ValueError(f"at most {self.max_prompts} prompts per forward pass (got {n_prompts})")
            if self.remaining <= 0:
                raise BudgetExhausted(f"forward-pass budget exhausted ({self.total} used)")
            self.used += 1
            return self.remaining

    def state(self) -> dict:
        return {"total": self.total, "used": self.used, "remaining": self.remaining,
                "max_prompts_per_pass": self.max_prompts}
