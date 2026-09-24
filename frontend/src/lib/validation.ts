import { z } from 'zod';

export const environmentalSchema = z.object({
  temperature_c: z
    .number()
    .min(-100, 'Min: -100°C')
    .max(60, 'Max: 60°C'),
  sea_surface_temperature_c: z
    .number()
    .min(-5, 'Min: -5°C')
    .max(45, 'Max: 45°C'),
  relative_humidity_pct: z
    .number()
    .min(0, 'Min: 0%')
    .max(100, 'Max: 100%'),
  water_vapour_gkg: z
    .number()
    .min(0, 'Min: 0 g/kg')
    .max(100, 'Max: 100 g/kg'),
  surface_pressure_hpa: z
    .number()
    .min(800, 'Min: 800 hPa')
    .max(1100, 'Max: 1100 hPa'),
  wind_speed_ms: z
    .number()
    .min(0, 'Min: 0 m/s')
    .max(120, 'Max: 120 m/s'),
  wind_direction_deg: z
    .number()
    .min(0, 'Min: 0°')
    .max(360, 'Max: 360°'),
  vertical_wind_shear_ms: z
    .number()
    .min(0, 'Min: 0 m/s')
    .max(50, 'Max: 50 m/s'),
  cloud_organization_index: z
    .number()
    .min(0, 'Min: 0')
    .max(1, 'Max: 1'),
  convection_index: z
    .number()
    .min(0, 'Min: 0')
    .max(1, 'Max: 1'),
  rotation_index: z
    .number()
    .min(0, 'Min: 0')
    .max(1, 'Max: 1'),
  persistence_index: z
    .number()
    .min(0, 'Min: 0')
    .max(1, 'Max: 1'),
  case_id: z.string().optional(),
});

export type EnvironmentalFormData = z.infer<typeof environmentalSchema>;
