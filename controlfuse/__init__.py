"""Official PyTorch implementation of ControlFuse (AAAI 2026)."""

__all__ = ["ControlFuse", "ControlFuseCriterion", "build_model"]


def __getattr__(name):
    if name in {"ControlFuse", "build_model"}:
        from .model import ControlFuse, build_model
        return {"ControlFuse": ControlFuse, "build_model": build_model}[name]
    if name == "ControlFuseCriterion":
        from .losses import ControlFuseCriterion
        return ControlFuseCriterion
    raise AttributeError(name)
