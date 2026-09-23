"""Pydantic schemas for prediction input data."""

from __future__ import annotations

from typing import Any
from fastapi import Form
from pydantic import BaseModel, Field


class NumericalFeaturesInput(BaseModel):
    """Atmospheric numerical environmental features required by CycloSense ML models."""

    temperature_c: float = Field(
        ...,
        ge=-100.0,
        le=60.0,
        description="Ambient air temperature in degrees Celsius [-100.0, 60.0]",
        examples=[28.5],
    )
    sea_surface_temperature_c: float = Field(
        ...,
        ge=-5.0,
        le=45.0,
        description="Sea surface temperature in degrees Celsius [-5.0, 45.0]",
        examples=[30.5],
    )
    relative_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity percentage [0.0, 100.0]",
        examples=[88.0],
    )
    water_vapour_gkg: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Water vapour mixing ratio in g/kg [0.0, 100.0]",
        examples=[22.0],
    )
    surface_pressure_hpa: float = Field(
        ...,
        ge=800.0,
        le=1100.0,
        description="Atmospheric surface pressure in hPa [800.0, 1100.0]",
        examples=[975.0],
    )
    wind_speed_ms: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Sustained surface wind speed in m/s [0.0, 120.0]",
        examples=[42.0],
    )
    wind_direction_deg: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Wind direction azimuth in degrees [0.0, 360.0]",
        examples=[120.0],
    )
    vertical_wind_shear_ms: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Vertical wind shear in m/s [0.0, 50.0]",
        examples=[4.5],
    )
    cloud_organization_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized cloud canopy organization index [0.0, 1.0]",
        examples=[0.92],
    )
    convection_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized deep convective activity index [0.0, 1.0]",
        examples=[0.88],
    )
    rotation_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized low-level vorticity / rotation index [0.0, 1.0]",
        examples=[0.85],
    )
    persistence_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized temporal disturbance persistence index [0.0, 1.0]",
        examples=[0.90],
    )

    def to_features_dict(self) -> dict[str, float]:
        """Convert validated schema into flat dictionary expected by NumericalPipeline."""
        return {
            "temperature_c": float(self.temperature_c),
            "sea_surface_temperature_c": float(self.sea_surface_temperature_c),
            "relative_humidity_pct": float(self.relative_humidity_pct),
            "water_vapour_gkg": float(self.water_vapour_gkg),
            "surface_pressure_hpa": float(self.surface_pressure_hpa),
            "wind_speed_ms": float(self.wind_speed_ms),
            "wind_direction_deg": float(self.wind_direction_deg),
            "vertical_wind_shear_ms": float(self.vertical_wind_shear_ms),
            "cloud_organization_index": float(self.cloud_organization_index),
            "convection_index": float(self.convection_index),
            "rotation_index": float(self.rotation_index),
            "persistence_index": float(self.persistence_index),
        }

    @classmethod
    def as_form(
        cls,
        temperature_c: float = Form(..., description="Ambient air temperature in °C [-100.0, 60.0]"),
        sea_surface_temperature_c: float = Form(..., description="Sea surface temperature in °C [-5.0, 45.0]"),
        relative_humidity_pct: float = Form(..., description="Relative humidity % [0.0, 100.0]"),
        water_vapour_gkg: float = Form(..., description="Water vapour mixing ratio in g/kg [0.0, 100.0]"),
        surface_pressure_hpa: float = Form(..., description="Surface pressure in hPa [800.0, 1100.0]"),
        wind_speed_ms: float = Form(..., description="Wind speed in m/s [0.0, 120.0]"),
        wind_direction_deg: float = Form(..., description="Wind direction in degrees [0.0, 360.0]"),
        vertical_wind_shear_ms: float = Form(..., description="Vertical wind shear in m/s [0.0, 50.0]"),
        cloud_organization_index: float = Form(..., description="Cloud organization index [0.0, 1.0]"),
        convection_index: float = Form(..., description="Convection index [0.0, 1.0]"),
        rotation_index: float = Form(..., description="Rotation index [0.0, 1.0]"),
        persistence_index: float = Form(..., description="Persistence index [0.0, 1.0]"),
    ) -> NumericalFeaturesInput:
        from pydantic import ValidationError as PydanticValidationError
        from fastapi import HTTPException, status

        try:
            return cls(
                temperature_c=temperature_c,
                sea_surface_temperature_c=sea_surface_temperature_c,
                relative_humidity_pct=relative_humidity_pct,
                water_vapour_gkg=water_vapour_gkg,
                surface_pressure_hpa=surface_pressure_hpa,
                wind_speed_ms=wind_speed_ms,
                wind_direction_deg=wind_direction_deg,
                vertical_wind_shear_ms=vertical_wind_shear_ms,
                cloud_organization_index=cloud_organization_index,
                convection_index=convection_index,
                rotation_index=rotation_index,
                persistence_index=persistence_index,
            )
        except PydanticValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(),
            )
