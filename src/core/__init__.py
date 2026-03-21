"""src/core — canonical schema and standardised model objects."""
from src.core.data_contract import (
    DataContract,
    DataContractError,
    StandardizedModelObject,
    validate_split_label,
)

__all__ = [
    "DataContract",
    "DataContractError",
    "StandardizedModelObject",
    "validate_split_label",
]
