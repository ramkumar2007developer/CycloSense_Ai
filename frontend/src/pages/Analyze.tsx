import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Activity, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';

import { environmentalSchema, type EnvironmentalFormData } from '@/lib/validation';
import { usePrediction } from '@/hooks/usePrediction';
import ImageUploader from '@/components/ImageUploader';
import EnvironmentalForm from '@/components/EnvironmentalForm';
import RiskGauge from '@/components/RiskGauge';
import SignalBreakdown from '@/components/SignalBreakdown';
import VerificationStatus from '@/components/VerificationStatus';
import ExplanationPanel from '@/components/ExplanationPanel';
import LoadingAnalysis from '@/components/LoadingAnalysis';
import RadarHUD from '@/components/RadarHUD';

export default function Analyze() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string>('');
  const [showRawJson, setShowRawJson] = useState(false);
  
  const methods = useForm<EnvironmentalFormData>({
    resolver: zodResolver(environmentalSchema),
    mode: 'onTouched',
    defaultValues: {
      temperature_c: 28.5,
      sea_surface_temperature_c: 30.2,
      relative_humidity_pct: 85,
      water_vapour_gkg: 18.5,
      surface_pressure_hpa: 1008,
      wind_speed_ms: 45,
      wind_direction_deg: 270,
      vertical_wind_shear_ms: 8,
      cloud_organization_index: 0.72,
      convection_index: 0.68,
      rotation_index: 0.55,
      persistence_index: 0.8,
    }
  });

  const { state, predict, reset } = usePrediction();

  const onSubmit = async (data: EnvironmentalFormData) => {
    if (!imageFile) {
      setImageError('Please upload a satellite image before analyzing.');
      return;
    }
    setImageError('');
    await predict(imageFile, data);
  };

  const handleReset = () => {
    setImageFile(null);
    setImageError('');
    setShowRawJson(false);
    methods.reset();
    reset();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (state.status === 'loading') {
    return <LoadingAnalysis />;
  }

  return (
    <main className="min-h-screen pt-24 pb-20 px-6 relative z-10">
      
      {/* Radar HUD overlay */}
      {(state.status === 'idle' || state.status === 'success') && (
        <RadarHUD 
          isAnalyzing={false} 
          riskLevel={state.status === 'success' ? state.data.prediction.development_risk_index : undefined} 
        />
      )}

      <div className="max-w-7xl mx-auto relative z-10">
        <AnimatePresence mode="wait">
          {state.status === 'idle' && (
            <motion.div
              key="form"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12"
            >
              {/* Left Column: Image & Instructions */}
              <div className="lg:col-span-5 flex flex-col gap-8">
                <div>
                  <h1 className="text-3xl sm:text-4xl font-black text-white mb-3 tracking-tight">
                    MISSION CONTROL
                  </h1>
                  <p className="text-xs tracking-wider text-cyan-400/80 uppercase font-mono">
                    System Ready // Awaiting Input
                  </p>
                </div>
                
                <div className="glass-panel p-6 rounded-2xl">
                  <ImageUploader 
                    onImageChange={(file) => {
                      setImageFile(file);
                      if (file) setImageError('');
                    }}
                    error={imageError}
                  />
                </div>

                <div className="glass-panel p-6 rounded-2xl">
                  <div className="flex items-start gap-3">
                    <Activity size={18} className="text-cyan-400 shrink-0 mt-0.5" />
                    <p className="text-[11px] text-gray-400 uppercase tracking-widest leading-relaxed">
                      Initialize multimodal fusion analysis by providing visual and environmental vectors.
                    </p>
                  </div>
                </div>
              </div>

              {/* Right Column: Environmental Parameters Form */}
              <div className="lg:col-span-7">
                <div className="glass-panel p-6 sm:p-8 rounded-2xl">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-sm font-bold tracking-widest text-white uppercase">Environmental Parameters</h2>
                      <p className="text-[10px] uppercase font-mono tracking-widest text-cyan-400/60 mt-2">
                        SURFACE LEVEL // 850 hPa
                      </p>
                    </div>
                  </div>

                  <FormProvider {...methods}>
                    <form onSubmit={methods.handleSubmit(onSubmit)} className="flex flex-col gap-8">
                      <EnvironmentalForm />
                      
                      <div className="pt-4 border-t border-white/10">
                        <button
                          type="submit"
                          className="btn-inference w-full"
                          disabled={methods.formState.isSubmitting}
                        >
                          <Cpu size={18} />
                          {methods.formState.isSubmitting ? 'INITIALIZING...' : 'RUN AI ANALYSIS'}
                        </button>
                        {Object.keys(methods.formState.errors).length > 0 && (
                          <p className="text-red-400 text-xs text-center mt-3 uppercase tracking-wider">
                            Resolve data anomalies before proceeding.
                          </p>
                        )}
                      </div>
                    </form>
                  </FormProvider>
                </div>
              </div>
            </motion.div>
          )}

          {state.status === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-2xl mx-auto glass-panel p-8 rounded-2xl text-center"
            >
              <div className="w-16 h-16 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto mb-6">
                <AlertTriangle size={32} className="text-red-400" />
              </div>
              <h2 className="text-lg tracking-widest uppercase font-bold text-white mb-3">Analysis Failed</h2>
              <p className="text-sm text-gray-400 mb-8 font-mono">
                {state.error.message}
              </p>
              <button onClick={handleReset} className="btn-ghost">
                REINITIALIZE SYSTEM
              </button>
            </motion.div>
          )}

          {state.status === 'success' && (
            <motion.div
              key="success"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8 }}
              className="flex flex-col gap-8"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <h1 className="text-3xl font-black text-white mb-2 tracking-tight uppercase">Analysis Complete</h1>
                  <div className="flex flex-wrap items-center gap-3 text-[10px] text-cyan-400/80 font-mono tracking-widest uppercase">
                    <span>{state.timestamp.toISOString().replace('T', ' ').slice(0, 19)} UTC</span>
                    {state.caseId && (
                      <>
                        <span className="text-cyan-400/30">|</span>
                        <span>ID: {state.caseId}</span>
                      </>
                    )}
                  </div>
                </motion.div>
                <motion.button 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  onClick={handleReset} 
                  className="btn-ghost text-[10px] py-2 px-4"
                >
                  NEW ANALYSIS
                </motion.button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column: Risk & Signals */}
                <div className="lg:col-span-5 flex flex-col gap-8">
                  <motion.div 
                    className="glass-panel p-6 sm:p-8 rounded-2xl"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.6 }}
                  >
                    <RiskGauge value={state.data.prediction.development_risk_index} />
                  </motion.div>
                  
                  <motion.div 
                    className="glass-panel p-6 sm:p-8 rounded-2xl"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.8 }}
                  >
                    <SignalBreakdown 
                      visualSignal={state.data.prediction.visual_signal}
                      environmentalSignal={state.data.prediction.environmental_signal}
                      fusionSignal={state.data.prediction.fusion_signal}
                    />
                  </motion.div>
                </div>

                {/* Right Column: Verification & Explanation */}
                <div className="lg:col-span-7 flex flex-col gap-8">
                  <motion.div 
                    className="glass-panel p-6 sm:p-8 rounded-2xl"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1.0 }}
                  >
                    <div className="flex flex-col gap-8">
                      <VerificationStatus 
                        status={state.data.verification.status} 
                        message={state.data.verification.message} 
                      />
                      
                      <div className="h-px w-full bg-cyan-400/10" />
                      
                      <ExplanationPanel explanation={state.data.explanation} />
                    </div>
                  </motion.div>
                </div>
              </div>

              {/* Raw JSON viewer */}
              <motion.div 
                className="mt-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
              >
                <button 
                  onClick={() => setShowRawJson(!showRawJson)}
                  className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-gray-500 hover:text-cyan-400 transition-colors mx-auto"
                >
                  {showRawJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  VIEW RAW RESPONSE
                </button>
                <AnimatePresence>
                  {showRawJson && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden mt-6"
                    >
                      <pre className="glass-panel p-6 rounded-xl text-[11px] font-mono text-cyan-400/70 overflow-x-auto">
                        {JSON.stringify(state.data, null, 2)}
                      </pre>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
