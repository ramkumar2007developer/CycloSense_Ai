import { motion } from 'framer-motion';
import { Satellite, BarChart3, BrainCircuit, ShieldCheck } from 'lucide-react';

const FEATURES = [
  {
    icon: Satellite,
    title: 'Satellite Imagery Analysis',
    desc: 'Detects cyclone patterns from high-resolution satellite images using advanced computer vision.',
    color: '#22d3ee',
  },
  {
    icon: BarChart3,
    title: 'Environmental Data',
    desc: 'Uses 12 atmospheric and oceanic parameters for comprehensive prediction accuracy.',
    color: '#3b82f6',
  },
  {
    icon: BrainCircuit,
    title: 'Advanced ML/DL Models',
    desc: 'Combines CNN, MLP and fusion models for multi-source multimodal signal analysis.',
    color: '#8b5cf6',
  },
  {
    icon: ShieldCheck,
    title: 'LLM Verification & Explanation',
    desc: 'Provides secondary visual verification and clear, human-readable meteorological insights.',
    color: '#22d3ee',
  },
];

export default function FeatureStrip() {
  return (
    <section className="relative z-10 -mt-1 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="rounded-2xl overflow-hidden"
          style={{
            background: 'rgba(255,255,255,0.025)',
            border: '1px solid rgba(255,255,255,0.08)',
            backdropFilter: 'blur(16px)',
          }}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((f, i) => {
              const Icon = f.icon;
              return (
                <motion.div
                  key={f.title}
                  className="relative p-8 flex flex-col gap-4 group"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                >
                  {/* Vertical divider between columns */}
                  {i < FEATURES.length - 1 && (
                    <div
                      className="hidden lg:block absolute right-0 top-8 bottom-8 w-px"
                      style={{ background: 'rgba(255,255,255,0.06)' }}
                    />
                  )}

                  {/* Icon */}
                  <div
                    className="w-11 h-11 rounded-xl flex items-center justify-center transition-transform group-hover:scale-110"
                    style={{
                      background: `rgba(${f.color === '#22d3ee' ? '34,211,238' : f.color === '#3b82f6' ? '59,130,246' : '139,92,246'},0.1)`,
                      border: `1px solid rgba(${f.color === '#22d3ee' ? '34,211,238' : f.color === '#3b82f6' ? '59,130,246' : '139,92,246'},0.2)`,
                    }}
                  >
                    <Icon size={20} style={{ color: f.color }} strokeWidth={1.5} />
                  </div>

                  {/* Text */}
                  <div>
                    <h3 className="font-semibold text-white text-sm mb-2 leading-snug">
                      {f.title}
                    </h3>
                    <p className="text-xs text-gray-500 leading-relaxed">{f.desc}</p>
                  </div>

                  {/* Hover accent line */}
                  <div
                    className="absolute bottom-0 left-8 right-8 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${f.color}, transparent)` }}
                  />
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
