import { motion } from 'framer-motion';

export default function RadarHUD({ isAnalyzing, riskLevel }: { isAnalyzing: boolean, riskLevel?: number }) {
  return (
    <div className="fixed inset-0 pointer-events-none z-[1] overflow-hidden">
      {/* Targeter Rings */}
      <div className="absolute top-1/2 left-[70%] -translate-y-1/2 -translate-x-1/2">
        <motion.div 
          className="relative w-[600px] h-[600px] rounded-full border border-cyan-400/10 flex items-center justify-center"
          animate={{ scale: [1, 1.05, 1], opacity: [0.15, 0.25, 0.15] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        >
          <div className="absolute w-[400px] h-[400px] rounded-full border border-cyan-400/20" />
          <div className="absolute w-[200px] h-[200px] rounded-full border border-cyan-400/30" />
          <div className="absolute w-2 h-2 rounded-full bg-cyan-400/50" />
          
          {/* Crosshair Lines */}
          <div className="absolute w-full h-[1px] bg-cyan-400/20" />
          <div className="absolute h-full w-[1px] bg-cyan-400/20" />
          
          {/* Scanning Line (when analyzing) */}
          {isAnalyzing && (
            <motion.div 
              className="absolute top-0 left-0 w-full h-[2px] bg-cyan-400 shadow-[0_0_15px_#22d3ee]"
              animate={{ y: [0, 600, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
            />
          )}
        </motion.div>
      </div>

      {/* Floating Panel 1 */}
      <motion.div 
        className="absolute top-[30%] left-[55%] glass-panel-strong px-4 py-3 rounded text-[10px] uppercase font-mono tracking-widest text-cyan-400/80"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        {isAnalyzing ? (
          <span className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse" />
            ANALYZING...
          </span>
        ) : riskLevel !== undefined ? (
          <div className="flex flex-col gap-1">
            <span className="text-white">CYCLONE DETECTED</span>
            <span className={riskLevel > 64 ? 'text-red-400' : riskLevel > 39 ? 'text-amber-400' : 'text-cyan-400'}>
              {riskLevel > 64 ? 'High' : riskLevel > 39 ? 'Moderate' : 'Low'} Risk · {riskLevel.toFixed(1)} / 100
            </span>
          </div>
        ) : (
          <span className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full" />
            SYSTEM READY<br/>
            AWAITING ANALYSIS
          </span>
        )}
      </motion.div>

      {/* Coordinates */}
      <div className="absolute bottom-[20%] right-[10%] text-[9px] font-mono text-cyan-400/30 tracking-widest text-right">
        <div>LAT: 15.2N</div>
        <div>LON: 85.1E</div>
        <div>ALT: 850 hPa</div>
      </div>
    </div>
  );
}
