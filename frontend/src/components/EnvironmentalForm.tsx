import { useFormContext } from 'react-hook-form';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { EnvironmentalFormData } from '@/lib/validation';

interface FieldConfig {
  name: keyof EnvironmentalFormData;
  label: string;
  unit: string;
  range: string;
  placeholder: string;
  step?: string;
}

const FIELD_GROUPS: { groupLabel: string; fields: FieldConfig[] }[] = [
  {
    groupLabel: 'Temperature',
    fields: [
      {
        name: 'temperature_c',
        label: 'Temperature',
        unit: '°C',
        range: '-100 to 60',
        placeholder: 'e.g. 28.5',
      },
      {
        name: 'sea_surface_temperature_c',
        label: 'Sea Surface Temperature',
        unit: '°C',
        range: '-5 to 45',
        placeholder: 'e.g. 30.2',
      },
    ],
  },
  {
    groupLabel: 'Moisture',
    fields: [
      {
        name: 'relative_humidity_pct',
        label: 'Relative Humidity',
        unit: '%',
        range: '0 to 100',
        placeholder: 'e.g. 85',
      },
      {
        name: 'water_vapour_gkg',
        label: 'Water Vapour',
        unit: 'g/kg',
        range: '0 to 100',
        placeholder: 'e.g. 18.5',
      },
    ],
  },
  {
    groupLabel: 'Atmospheric',
    fields: [
      {
        name: 'surface_pressure_hpa',
        label: 'Surface Pressure',
        unit: 'hPa',
        range: '800 to 1100',
        placeholder: 'e.g. 1008',
      },
      {
        name: 'wind_speed_ms',
        label: 'Wind Speed',
        unit: 'm/s',
        range: '0 to 120',
        placeholder: 'e.g. 45',
      },
      {
        name: 'wind_direction_deg',
        label: 'Wind Direction',
        unit: '°',
        range: '0 to 360',
        placeholder: 'e.g. 270',
      },
      {
        name: 'vertical_wind_shear_ms',
        label: 'Vertical Wind Shear',
        unit: 'm/s',
        range: '0 to 50',
        placeholder: 'e.g. 8',
      },
    ],
  },
  {
    groupLabel: 'Cyclone Indices',
    fields: [
      {
        name: 'cloud_organization_index',
        label: 'Cloud Organization Index',
        unit: '0–1',
        range: '0 to 1',
        placeholder: 'e.g. 0.72',
        step: '0.01',
      },
      {
        name: 'convection_index',
        label: 'Convection Index',
        unit: '0–1',
        range: '0 to 1',
        placeholder: 'e.g. 0.68',
        step: '0.01',
      },
      {
        name: 'rotation_index',
        label: 'Rotation Index',
        unit: '0–1',
        range: '0 to 1',
        placeholder: 'e.g. 0.55',
        step: '0.01',
      },
      {
        name: 'persistence_index',
        label: 'Persistence Index',
        unit: '0–1',
        range: '0 to 1',
        placeholder: 'e.g. 0.80',
        step: '0.01',
      },
    ],
  },
];

function FieldGroup({ groupLabel, fields, defaultOpen = true }: {
  groupLabel: string;
  fields: FieldConfig[];
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const { register, formState: { errors } } = useFormContext<EnvironmentalFormData>();

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: '1px solid rgba(255,255,255,0.07)' }}
    >
      {/* Group header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors hover:bg-white/03"
        style={{ background: 'rgba(255,255,255,0.03)' }}
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: '#22d3ee', boxShadow: '0 0 6px rgba(34,211,238,0.6)' }}
          />
          <span className="text-xs font-bold tracking-widest uppercase text-gray-400">
            {groupLabel}
          </span>
        </div>
        {open ? (
          <ChevronUp size={14} className="text-gray-600" />
        ) : (
          <ChevronDown size={14} className="text-gray-600" />
        )}
      </button>

      {/* Fields */}
      {open && (
        <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {fields.map((field) => {
            const error = errors[field.name];
            return (
              <div key={field.name}>
                {/* Label row */}
                <div className="flex items-center justify-between mb-1.5">
                  <label
                    htmlFor={field.name}
                    className="text-xs font-medium text-gray-300"
                  >
                    {field.label}
                  </label>
                  <span className="text-xs font-mono text-gray-600">{field.unit}</span>
                </div>

                {/* Input */}
                <input
                  id={field.name}
                  type="number"
                  step={field.step ?? 'any'}
                  placeholder={field.placeholder}
                  className={`cyclo-input ${error ? 'error' : ''}`}
                  aria-invalid={!!error}
                  aria-describedby={error ? `${field.name}-error` : `${field.name}-hint`}
                  {...register(field.name, { valueAsNumber: true })}
                />

                {/* Hint / error */}
                {error ? (
                  <p
                    id={`${field.name}-error`}
                    className="text-xs text-red-400 mt-1"
                    role="alert"
                  >
                    {error.message as string}
                  </p>
                ) : (
                  <p
                    id={`${field.name}-hint`}
                    className="text-xs text-gray-600 mt-1 font-mono"
                  >
                    Range: {field.range}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function EnvironmentalForm() {
  const { register } = useFormContext<EnvironmentalFormData>();

  return (
    <div className="flex flex-col gap-3">
      {/* Case ID */}
      <div
        className="rounded-xl p-4"
        style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
      >
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="case_id" className="text-xs font-medium text-gray-300">
            Case ID
            <span className="text-gray-600 font-normal ml-1">(optional)</span>
          </label>
        </div>
        <input
          id="case_id"
          type="text"
          placeholder="e.g. BAY-2026-001"
          className="cyclo-input"
          {...register('case_id')}
        />
      </div>

      {/* Parameter groups */}
      {FIELD_GROUPS.map((group, i) => (
        <FieldGroup
          key={group.groupLabel}
          groupLabel={group.groupLabel}
          fields={group.fields}
          defaultOpen={i === 0}
        />
      ))}
    </div>
  );
}
