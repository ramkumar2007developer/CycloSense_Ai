import { motion } from 'framer-motion';
import { Upload, Activity, Cpu, ShieldCheck } from 'lucide-react';

const STEPS = [
  {
    num: '01',
    icon: Upload,
    title: 'UPLOAD',
    desc: 'Upload a satellite imagery frame (JPG/PNG) for cyclone analysis. The system accepts high-resolution IR or visible channel imagery.',
    color: '#22d3ee',
  },
  {
    num: '02',
    icon: Activity,
    title: 'ANALYZE',
    desc: 'Enter 12 atmospheric and environmental parameters. The AI processes all signals simultaneously for comprehensive evaluation.',
    color: '#3b82f6',
  },
  {
    num: '03',
    icon: Cpu,
    title: 'PREDICT',
    desc: 'AI models calculate visual, environmental and fusion signals. A development/risk index and classification are produced.',
    color: '#8b5cf6',
  },
  {
    num: '04',
    icon: ShieldCheck,
    title: 'VERIFY',
    desc: 'LLM vision verification evaluates the prediction when required. Explainable reasoning is generated for meteorologists.',
    color: '#22d3ee',
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="relative z-10 py-32 px-6">
      {/* Background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 0%, rgba(8,20,40,0.4) 0%, transparent 60%)',
        }}
      />

      <div className="max-w-7xl mx-auto relative">
        <motion.div
          className="text-center mb-20"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className="label-sm text-cyan-400 block mb-4">Workflow</span>
          <h2 className="text-4xl sm:text-5xl font-black text-white">
            How It Works
          </h2>
        </motion.div>

        {/* Steps */}
        <div className="relative">
          {/* Connecting line (desktop) */}
          <div
            className="hidden lg:block absolute top-16 left-[12.5%] right-[12.5%] h-px z-0"
            style={{
              background: 'linear-gradient(90deg, rgba(34,211,238,0), rgba(34,211,238,0.3) 25%, rgba(59,130,246,0.3) 50%, rgba(139,92,246,0.3) 75%, rgba(34,211,238,0))',
            }}
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-4">
            {STEPS.map((step, i) => {
              const Icon = step.icon;
              return (
                <motion.div
                  key={step.num}
                  className="relative flex flex-col items-center text-center gap-5 p-6"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: i * 0.15 }}
                >
                  {/* Step circle */}
                  <div className="relative z-10">
                    <div
                      className="w-14 h-14 rounded-full flex items-center justify-center relative"
                      style={{
                        background: 'rgba(5,7,9,1)',
                        border: `1px solid ${step.color}40`,
                        boxShadow: `0 0 24px ${step.color}15`,
                      }}
                    >
                      <Icon size={22} style={{ color: step.color }} strokeWidth={1.5} />
                      {/* Pulsing ring */}
                      <div
                        className="absolute inset-0 rounded-full border"
                        style={{
                          borderColor: `${step.color}20`,
                          animation: 'pulse-ring 2.5s ease-out infinite',
                          animationDelay: `${i * 0.5}s`,
                        }}
                      />
                    </div>

                    {/* Number */}
                    <div
                      className="absolute -top-2 -right-2 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                      style={{
                        background: step.color,
                        color: '#050709',
                        fontSize: '10px',
                      }}
                    >
                      {i + 1}
                    </div>
                  </div>

                  {/* Step content */}
                  <div>
                    <div className="label-xs mb-2" style={{ color: step.color }}>
                      STEP {step.num}
                    </div>
                    <h3 className="text-lg font-bold text-white mb-3">{step.title}</h3>
                    <p className="text-xs text-gray-500 leading-relaxed">{step.desc}</p>
                  </div>

                  {/* Flow particles (animated dots on the line) */}
                  {i < STEPS.length - 1 && (
                    <div className="hidden lg:block absolute top-[56px] right-0 w-8 overflow-hidden">
                      {[0, 1, 2].map((d) => (
                        <div
                          key={d}
                          className="absolute w-1 h-1 rounded-full"
                          style={{
                            background: step.color,
                            animation: `flowRight 2s linear infinite`,
                            animationDelay: `${d * 0.66}s`,
                            opacity: 0.6,
                          }}
                        />
                      ))}
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* CTA */}
        <motion.div
          className="text-center mt-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <a href="/analyze" className="btn-primary inline-flex">
            Begin Analysis
          </a>
        </motion.div>
      </div>

      <style>{`
        @keyframes flowRight {
          0% { transform: translateX(-8px); opacity: 0; }
          20% { opacity: 0.8; }
          80% { opacity: 0.8; }
          100% { transform: translateX(32px); opacity: 0; }
        }
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(1.8); opacity: 0; }
        }
      `}</style>
    </section>
  );
}
