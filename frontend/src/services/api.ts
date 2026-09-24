import axios from 'axios';
import type { CycloNexisResponse, ApiError } from '@/types/cyclone';
import type { EnvironmentalFormData } from '@/lib/validation';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for AI inference
});

/**
 * Send multipart/form-data to the FastAPI /predict endpoint.
 * Attaches satellite image + all 12 environmental parameters.
 */
export async function predictCyclone(
  image: File,
  params: EnvironmentalFormData
): Promise<CycloNexisResponse> {
  const formData = new FormData();

  // Attach satellite image
  formData.append('image', image, image.name);

  // Attach all environmental parameters as numeric fields
  formData.append('temperature_c', String(params.temperature_c));
  formData.append('sea_surface_temperature_c', String(params.sea_surface_temperature_c));
  formData.append('relative_humidity_pct', String(params.relative_humidity_pct));
  formData.append('water_vapour_gkg', String(params.water_vapour_gkg));
  formData.append('surface_pressure_hpa', String(params.surface_pressure_hpa));
  formData.append('wind_speed_ms', String(params.wind_speed_ms));
  formData.append('wind_direction_deg', String(params.wind_direction_deg));
  formData.append('vertical_wind_shear_ms', String(params.vertical_wind_shear_ms));
  formData.append('cloud_organization_index', String(params.cloud_organization_index));
  formData.append('convection_index', String(params.convection_index));
  formData.append('rotation_index', String(params.rotation_index));
  formData.append('persistence_index', String(params.persistence_index));

  // Optional case ID
  if (params.case_id) {
    formData.append('case_id', params.case_id);
  }

  try {
    const response = await apiClient.post<CycloNexisResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (!error.response) {
        const apiError: ApiError = {
          type: 'network',
          message: 'Unable to connect to the CycloNexis AI backend. Please check your connection and ensure the server is running.',
        };
        throw apiError;
      }

      const status = error.response.status;

      if (status === 422) {
        const apiError: ApiError = {
          type: 'validation',
          message: 'Some input values are invalid. Please review the highlighted fields and correct any errors.',
        };
        throw apiError;
      }

      if (status === 503) {
        const apiError: ApiError = {
          type: 'unavailable',
          message: 'The AI prediction service is temporarily unavailable. Please try again in a few moments.',
        };
        throw apiError;
      }

      // 500 and other server errors
      const apiError: ApiError = {
        type: 'server',
        message: 'The prediction service encountered an unexpected error. Please try again.',
      };
      throw apiError;
    }

    // Unknown error
    const apiError: ApiError = {
      type: 'network',
      message: 'An unexpected error occurred. Please try again.',
    };
    throw apiError;
  }
}
