"""Project-specific exceptions."""


class CycloSenseError(Exception):
    """Base exception."""


class ConfigurationError(CycloSenseError):
    """Invalid configuration."""


class DatasetError(CycloSenseError):
    """Dataset loading or validation error."""


class DatasetValidationError(DatasetError):
    """Dataset contents failed validation."""


class ModelError(CycloSenseError):
    """Model training or inference error."""


class ValidationError(CycloSenseError):
    """Input validation error."""
