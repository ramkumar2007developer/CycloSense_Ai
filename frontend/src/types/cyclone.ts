// CycloNexis AI — TypeScript Interfaces

export interface EnvironmentalParams {
  temperature_c: number;
  sea_surface_temperature_c: number;
  relative_humidity_pct: number;
  water_vapour_gkg: number;
  surface_pressure_hpa: number;
  wind_speed_ms: number;
  wind_direction_deg: number;
  vertical_wind_shear_ms: number;
  cloud_organization_index: number;
  convection_index: number;
  rotation_index: number;
  persistence_index: number;
}

export interface PredictionFormInput extends EnvironmentalParams {
  case_id?: string;
  image: File;
}

export interface PredictionResult {
  classification: string;
  development_risk_index: number;
  visual_signal: number;
  environmental_signal: number;
  fusion_signal: number;
}

export interface VerificationResult {
  status: string;
  message?: string;
}

export interface ExplanationResult {
  summary: string;
  primary_drivers: string[];
  inhibiting_barriers: string[];
  operational_guidance: string;
}

export interface CycloNexisResponse {
  prediction: PredictionResult;
  verification: VerificationResult;
  explanation: ExplanationResult;
}

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH';

export function getRiskLevel(index: number): RiskLevel {
  if (index < 40) return 'LOW';
  if (index < 65) return 'MODERATE';
  return 'HIGH';
}

export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case 'LOW': return '#22d3ee'; // cyan
    case 'MODERATE': return '#f59e0b'; // amber
    case 'HIGH': return '#ef4444'; // red
  }
}

export const VERIFICATION_MESSAGES: Record<string, { title: string; description: string; color: string }> = {
  VERIFIED_CYCLONE: {
    title: 'Verified Cyclone',
    description: 'Visual verification supports the model prediction. LLM analysis confirms cyclone signatures.',
    color: '#ef4444',
  },
  LLM_NOT_REQUIRED: {
    title: 'LLM Not Required',
    description: 'Prediction confidence was sufficient without secondary LLM verification.',
    color: '#22d3ee',
  },
  HUMAN_REVIEW_REQUIRED: {
    title: 'Human Review Required',
    description: 'Model output requires human meteorological review. Confidence thresholds not met.',
    color: '#f59e0b',
  },
};

export type ApiError = {
  type: 'validation' | 'server' | 'unavailable' | 'network';
  message: string;
};
