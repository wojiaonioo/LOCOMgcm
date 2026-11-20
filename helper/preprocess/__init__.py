from .adapters import detect_adapter

__all__ = ["run_pipeline", "detect_adapter"]


def __getattr__(name):
    # Imported on first use: the pipeline pulls in the comparison metrics and
    # their xskillscore dependency, which the LOCOMgcm-only stage 4 route
    # (helper.preprocess.model_only) does not need.
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
