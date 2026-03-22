"""src/adapters — model-class-specific input adapters."""
from src.adapters.point_forecast import (
    Adapter_PointForecast,
    AdapterError,
    ResidualPool,
    bucket_coarse_4,
    bucket_hourly_24,
    bucket_none,
)

__all__ = [
    "Adapter_PointForecast",
    "AdapterError",
    "ResidualPool",
    "bucket_coarse_4",
    "bucket_hourly_24",
    "bucket_none",
]

from src.adapters.simulation_joint import (
    Adapter_SimulationJoint,
    JointSimulationObject,
    MarginalSamples,
    SimulationAdapterError,
)

from src.adapters.quantile_adapter import (
    Adapter_Quantiles,
    QuantileAdapterError,
    QuantileFunctionObject,
)

from src.adapters.build_dist_from_residuals import BuildDist_FromResiduals, SampleMatrix