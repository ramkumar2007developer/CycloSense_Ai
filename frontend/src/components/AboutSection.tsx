import { motion } from 'framer-motion';
import { Eye, Layers, Cpu, ShieldCheck } from 'lucide-react';

const STATS = [
  { value: '12', label: 'Environmental Parameters' },
  { value: '3', label: 'AI Signal Components' },
  { value: '100', label: 'Risk Index Scale' },
  { value: 'Multi-Modal', label: 'Analysis Engine' },
];

const PIPELINE = [
  { icon: Eye, label: 'Satellite Vision', color: '#22d3ee' },
  { icon: Layers, label: 'Environmental Intelligence', color: '#3b82f6' },
  { icon: Cpu, label: 'Multimodal AI', color: '#8b5cf6' },
  { icon: ShieldCheck, label: 'LLM Verification', color: '#22d3ee' },
];

export default function AboutSection() {
  return (
    <section id="about" className="relative z-10 py-32 px-6">
      {/* Background glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, rgba(8,20,40,0.5) 0%, transparent 70%)',
        }}
      />

      <div className="max-w-7xl mx-auto relative">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

          {/* Left: Text */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
          >
            <span className="label-sm text-cyan-400 mb-4 block">About the Platform</span>
            <h2 className="text-4xl sm:text-5xl font-black text-white leading-tight mb-6">
              Understanding the Storm{' '}
              <span className="text-gray-500">Before It Escalates.</span>
            </h2>
            <p className="text-gray-400 text-base leading-relaxed mb-8">
              CycloNexis AI is a multimodal meteorological intelligence system built to provide 
              early, accurate, and explainable cyclone detection. By fusing satellite imagery 
              analysis with 12 key atmospheric and oceanic indicators, our platform delivers 
              risk assessments that meteorologists and researchers can act on.
            </p>
            <p className="text-gray-500 text-sm leading-relaxed">
              Our AI pipeline combines deep learning computer vision models with environmental 
              signal processing and LLM-based secondary verification — creating a system that 
              not only predicts, but explains its reasoning in human-readable meteorological terms.
            </p>

            {/* Pipeline visual */}
            <div className="mt-10">
              <div className="flex flex-wrap items-center gap-3">
                {PIPELINE.map((step, i) => {
                  const Icon = step.icon;
                  return (
                    <div key={step.label} className="flex items-center gap-3">
                      <div
                        className="flex items-center gap-2 px-3 py-2 rounded-lg"
                        style={{
                          background: 'rgba(255,255,255,0.04)',
                          border: '1px solid rgba(255,255,255,0.08)',
                        }}
                      >
                        <Icon size={14} style={{ color: step.color }} strokeWidth={1.5} />
                        <span className="text-xs font-medium text-gray-300">{step.label}</span>
                      </div>
                      {i < PIPELINE.length - 1 && (
                        <span className="text-gray-600 text-xs">+</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>

          {/* Right: Stats grid */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="grid grid-cols-2 gap-4"
          >
            {STATS.map((stat, i) => (
              <motion.div
                key={stat.label}
                className="glass-panel rounded-2xl p-8 relative overflow-hidden group"
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 * i }}
                whileHover={{ borderColor: 'rgba(34,211,238,0.2)' }}
              >
                {/* Corner accent */}
                <div
                  className="absolute top-0 right-0 w-16 h-16 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{
                    background: 'radial-gradient(ellipse at top right, rgba(34,211,238,0.08), transparent)',
                  }}
                />
                <div className="text-3xl sm:text-4xl font-black text-white mb-2">{stat.value}</div>
                <div className="text-xs text-gray-500 uppercase tracking-wider">{stat.label}</div>
                {/* Bottom accent line */}
                <div
                  className="absolute bottom-0 left-6 right-6 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: 'linear-gradient(90deg, transparent, rgba(34,211,238,0.4), transparent)' }}
                />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
