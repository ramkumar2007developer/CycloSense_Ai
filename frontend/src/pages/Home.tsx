import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Mouse, ArrowRight, Satellite, Database, BrainCircuit, ShieldCheck } from 'lucide-react';

const FEATURES = [
  {
    icon: Satellite,
    title: 'SATELLITE IMAGERY ANALYSIS',
    desc: 'Deep learning vision models trained on tropical cyclone patterns.'
  },
  {
    icon: Database,
    title: 'ENVIRONMENTAL DATA',
    desc: '12 atmospheric and oceanic conditions evaluated in real-time.'
  },
  {
    icon: BrainCircuit,
    title: 'ADVANCED ML/DL MODELS',
    desc: 'Multimodal fusion architecture for higher prediction accuracy.'
  },
  {
    icon: ShieldCheck,
    title: 'LLM VERIFICATION & EXPLANATION',
    desc: 'Visual language models perform secondary validation.'
  }
];

export default function Home() {
  return (
    <main className="relative min-h-screen flex flex-col justify-between pt-32 pb-12 px-6">
      <div className="max-w-7xl mx-auto w-full flex-1 flex flex-col justify-center">
        
        <div className="max-w-2xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          >
            <span className="inline-block text-[10px] font-bold tracking-[0.2em] text-cyan-400 mb-6 uppercase">
              Smarter Insights. Safer Tomorrows.
            </span>
          </motion.div>

          <motion.h1 
            className="text-6xl sm:text-7xl lg:text-8xl font-black text-white leading-[1.1] mb-6 tracking-tight"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2, ease: 'easeOut' }}
          >
            CycloNexis AI
          </motion.h1>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4, ease: 'easeOut' }}
          >
            <h2 className="text-xl sm:text-2xl text-gray-300 font-light leading-snug mb-6">
              AI-Powered Tropical Cyclone <br className="hidden sm:block" />
              Prediction & Verification
            </h2>
          </motion.div>

          <motion.p 
            className="text-sm text-gray-500 leading-relaxed max-w-md mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6, ease: 'easeOut' }}
          >
            Combining satellite imagery, environmental data, multimodal AI analysis 
            and verification intelligence to detect and analyze tropical cyclone 
            development with greater explainability.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row gap-4 items-start"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8, ease: 'easeOut' }}
          >
            {/* Primary — GET STARTED */}
            <Link to="/analyze" className="btn-17">
              <span className="text-container">
                <span className="text">GET STARTED</span>
              </span>
            </Link>

            {/* Secondary — LEARN MORE */}
            <button
              className="btn-17 btn-17-outline"
              onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
            >
              <span className="text-container">
                <span className="text">LEARN MORE</span>
              </span>
            </button>
          </motion.div>
        </div>
      </div>

      {/* Bottom Feature Panel */}
      <motion.div 
        className="max-w-7xl mx-auto w-full mt-20"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 1.2, ease: 'easeOut' }}
      >
        <div className="glass-panel rounded-xl overflow-hidden grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-white/10">
          {FEATURES.map((feature, idx) => {
            const Icon = feature.icon;
            return (
              <div key={idx} className="p-6 group hover:bg-white/5 transition-colors cursor-default">
                <Icon size={20} className="text-cyan-400/70 mb-4 group-hover:text-cyan-400 group-hover:drop-shadow-[0_0_8px_rgba(34,211,238,0.5)] transition-all" />
                <h3 className="text-xs font-bold tracking-widest text-white mb-2 uppercase">{feature.title}</h3>
                <p className="text-xs text-gray-500 leading-relaxed group-hover:-translate-y-0.5 transition-transform">{feature.desc}</p>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* Scroll Indicator */}
      <motion.div 
        className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
      >
        <motion.div 
          animate={{ y: [0, 6, 0] }} 
          transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
        >
          <Mouse size={20} className="text-white/40" />
        </motion.div>
        <span className="text-[9px] font-bold tracking-[0.2em] text-white/40 uppercase">
          SCROLL TO EXPLORE
        </span>
      </motion.div>
    </main>
  );
}
