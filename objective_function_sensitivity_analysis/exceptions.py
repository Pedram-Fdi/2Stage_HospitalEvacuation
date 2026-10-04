"""Exceptions for objective-function sensitivity analysis."""


class SensitivityAnalysisError(Exception):
    """Base class for sensitivity-analysis errors."""


class InstanceNotFoundError(SensitivityAnalysisError):
    """Raised when the requested instance pickle cannot be loaded."""


class InvalidWeightConfigurationError(SensitivityAnalysisError):
    """Raised when policy weights are missing, negative, or malformed."""


class ModelConfigurationError(SensitivityAnalysisError):
    """Raised when the selected model/solver settings are incompatible."""


class OptimizationFailedError(SensitivityAnalysisError):
    """Raised when Gurobi does not return a usable solution."""


class ResultExtractionError(SensitivityAnalysisError):
    """Raised when solution metrics cannot be extracted."""


class ObjectiveValidationError(SensitivityAnalysisError):
    """Raised when recalculated objective components disagree with Gurobi."""
