import { motion } from 'framer-motion';
import { Eye, Layers, Gauge, ShieldCheck, MessageSquare, BarChart3 } from 'lucide-react';

const FEATURES = [
  {
    num: '01',
    icon: Eye,
    title: 'Satellite Image Intelligence',
    desc: 'Analyze uploaded satellite imagery using deep learning computer vision models trained on tropical cyclone patterns. Extract visual signals, eye formation, and banding structure.',
    wide: true,
  },
  {
    num: '02',
    icon: BarChart3,
    title: 'Environmental Analysis',
    desc: 'Evaluate 12 atmospheric and oceanic conditions including SST, wind shear, humidity, surface pressure, and cyclone organization indices.',
    wide: false,
  },
  {
    num: '03',
    icon: Layers,
    title: 'Multimodal Fusion',
    desc: 'Combine visual and numerical signals through a fusion architecture that weighs all input modalities for higher prediction accuracy.',
    wide: false,
  },
  {
    num: '04',
    icon: Gauge,
    title: 'Risk Index',
    desc: 'Generate a 0–100 development/risk score that communicates cyclone formation probability and intensity in a single, actionable metric.',
    wide: false,
  },
  {
    num: '05',
    icon: ShieldCheck,
    title: 'LLM Verification',
    desc: 'When model confidence requires secondary validation, a vision language model performs visual verification and flags ambiguous cases for human review.',
    wide: false,
  },
  {
    num: '06',
    icon: MessageSquare,
    title: 'Explainable AI',
    desc: 'Translate raw model outputs into readable meteorological reasoning — primary drivers, inhibiting barriers, and operational guidance. Science, clearly communicated.',
    wide: true,
  },
];

export default function FeaturesSection() {
  return (
    <section id="features" className="relative z-10 py-32 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <motion.div
          className="text-center mb-20"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className="label-sm text-cyan-400 block mb-4">Platform Capabilities</span>
          <h2 className="text-4xl sm:text-5xl font-black text-white leading-tight">
            One System.{' '}
            <span className="text-gray-500">Multiple Signals.</span>
            <br />
            Clearer Decisions.
          </h2>
        </motion.div>

        {/* Feature grid — asymmetric */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px"
          style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '16px', overflow: 'hidden' }}>
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <motion.div
                key={f.num}
                className="relative p-8 group"
                style={{ background: 'rgba(5,7,9,0.95)' }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
              >
                {/* Number */}
                <div className="text-6xl font-black text-white/04 absolute top-4 right-6 select-none leading-none">
                  {f.num}
                </div>

                {/* Icon */}
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center mb-5"
                  style={{
                    background: 'rgba(34,211,238,0.07)',
                    border: '1px solid rgba(34,211,238,0.15)',
                  }}
                >
                  <Icon size={18} className="text-cyan-400" strokeWidth={1.5} />
                </div>

                {/* Label */}
                <div className="label-xs text-cyan-400/60 mb-2">{f.num} —</div>

                {/* Title */}
                <h3 className="text-lg font-bold text-white mb-3 group-hover:text-cyan-50 transition-colors">
                  {f.title}
                </h3>

                {/* Desc */}
                <p className="text-sm text-gray-500 leading-relaxed">{f.desc}</p>

                {/* Hover border glow */}
                <div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
                  style={{
                    background: 'linear-gradient(135deg, rgba(34,211,238,0.03) 0%, transparent 60%)',
                    borderRadius: '16px',
                  }}
                />
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
