import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import RadarHUD from './RadarHUD';

const STAGES = [
  'PROCESSING SATELLITE IMAGERY',
  'EXTRACTING ENVIRONMENTAL SIGNALS',
  'RUNNING MULTIMODAL FUSION',
  'EVALUATING VERIFICATION GATE',
  'GENERATING EXPLANATION'
];

export default function LoadingAnalysis() {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    // Fake progress through the stages for visual effect (since real API might be fast)
    const interval = setInterval(() => {
      setCurrentStage(prev => Math.min(prev + 1, STAGES.length - 1));
    }, 800); // Progress every 800ms
    
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <RadarHUD isAnalyzing={true} />
      
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm pointer-events-none">
        <motion.div 
          className="glass-panel-strong p-8 rounded-xl flex flex-col gap-6 max-w-md w-full mx-4"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="flex flex-col gap-2">
            <h3 className="text-cyan-400 font-mono text-sm tracking-widest uppercase">
              Mission Control
            </h3>
            <div className="h-[1px] w-full bg-cyan-400/20" />
          </div>

          <div className="flex flex-col gap-4">
            {STAGES.map((stage, idx) => {
              const isPast = idx < currentStage;
              const isCurrent = idx === currentStage;
              
              return (
                <div key={stage} className="flex items-center gap-3">
                  <div className="w-4 flex justify-center">
                    {isPast ? (
                      <span className="text-cyan-400 text-xs">✓</span>
                    ) : isCurrent ? (
                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse" />
                    ) : (
                      <span className="w-1.5 h-1.5 bg-gray-600 rounded-full" />
                    )}
                  </div>
                  <span className={`text-[10px] font-mono tracking-widest ${
                    isPast ? 'text-cyan-400/50' : isCurrent ? 'text-white' : 'text-gray-600'
                  }`}>
                    0{idx + 1} // {stage}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </>
  );
}
