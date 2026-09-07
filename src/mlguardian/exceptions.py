"""Project-specific exceptions."""


class AuditError(Exception):
    """Base exception for audit failures."""


class DatasetLoadError(AuditError):
    """Raised when datasets cannot be loaded."""


class SchemaMismatchError(AuditError):
    """Raised when train/test schemas are incompatible."""


class ConfigurationError(AuditError):
    """Raised when configuration is invalid."""


class UnsupportedFormatError(DatasetLoadError):
    """Raised when the dataset format is unsupported."""
