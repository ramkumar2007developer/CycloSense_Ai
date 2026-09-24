import { useState, useCallback } from 'react';
import { predictCyclone } from '@/services/api';
import type { CycloNexisResponse, ApiError } from '@/types/cyclone';
import type { EnvironmentalFormData } from '@/lib/validation';

export type PredictionState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: CycloNexisResponse; caseId?: string; timestamp: Date }
  | { status: 'error'; error: ApiError };

export function usePrediction() {
  const [state, setState] = useState<PredictionState>({ status: 'idle' });

  const predict = useCallback(async (image: File, params: EnvironmentalFormData) => {
    setState({ status: 'loading' });
    try {
      const data = await predictCyclone(image, params);
      setState({
        status: 'success',
        data,
        caseId: params.case_id,
        timestamp: new Date(),
      });
      return data;
    } catch (err) {
      const apiError = err as ApiError;
      setState({ status: 'error', error: apiError });
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ status: 'idle' });
  }, []);

  return { state, predict, reset };
}
