import { motion } from 'framer-motion';
import { FileText, AlertCircle, Navigation, ChevronRight } from 'lucide-react';
import type { ExplanationResult } from '@/types/cyclone';

interface ExplanationPanelProps {
  explanation: ExplanationResult;
}

export default function ExplanationPanel({ explanation }: ExplanationPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="flex flex-col gap-5"
    >
      <div className="label-xs text-gray-500">AI METEOROLOGICAL EXPLANATION</div>

      {/* Summary */}
      <div
        className="rounded-xl p-5"
        style={{
          background: 'rgba(255,255,255,0.025)',
          border: '1px solid rgba(255,255,255,0.07)',
        }}
      >
        <div className="flex items-center gap-2 mb-3">
          <FileText size={13} className="text-cyan-400" strokeWidth={1.5} />
          <span className="text-xs font-bold tracking-widest uppercase text-cyan-400">Summary</span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">
          {explanation.summary || 'No summary provided.'}
        </p>
      </div>

      {/* Two column: drivers + barriers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Primary Drivers */}
        <div
          className="rounded-xl p-5"
          style={{
            background: 'rgba(239,68,68,0.04)',
            border: '1px solid rgba(239,68,68,0.12)',
          }}
        >
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle size={13} className="text-red-400" strokeWidth={1.5} />
            <span className="text-xs font-bold tracking-widest uppercase text-red-400">
              Primary Drivers
            </span>
          </div>
          {explanation.primary_drivers && explanation.primary_drivers.length > 0 ? (
            <ul className="flex flex-col gap-2" role="list">
              {explanation.primary_drivers.map((driver, i) => (
                <li key={i} className="flex items-start gap-2.5">
                  <ChevronRight size={12} className="text-red-400/60 mt-0.5 shrink-0" />
                  <span className="text-sm text-gray-400 leading-relaxed">{driver}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-600">No primary drivers identified.</p>
          )}
        </div>

        {/* Inhibiting Barriers */}
        <div
          className="rounded-xl p-5"
          style={{
            background: 'rgba(34,211,238,0.03)',
            border: '1px solid rgba(34,211,238,0.1)',
          }}
        >
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle size={13} className="text-cyan-400" strokeWidth={1.5} />
            <span className="text-xs font-bold tracking-widest uppercase text-cyan-400">
              Inhibiting Barriers
            </span>
          </div>
          {explanation.inhibiting_barriers && explanation.inhibiting_barriers.length > 0 ? (
            <ul className="flex flex-col gap-2" role="list">
              {explanation.inhibiting_barriers.map((barrier, i) => (
                <li key={i} className="flex items-start gap-2.5">
                  <ChevronRight size={12} className="text-cyan-400/60 mt-0.5 shrink-0" />
                  <span className="text-sm text-gray-400 leading-relaxed">{barrier}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-600">No inhibiting barriers identified.</p>
          )}
        </div>
      </div>

      {/* Operational Guidance */}
      <div
        className="rounded-xl p-5"
        style={{
          background: 'rgba(245,158,11,0.04)',
          border: '1px solid rgba(245,158,11,0.12)',
        }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Navigation size={13} className="text-amber-400" strokeWidth={1.5} />
          <span className="text-xs font-bold tracking-widest uppercase text-amber-400">
            Operational Guidance
          </span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">
          {explanation.operational_guidance || 'No operational guidance provided.'}
        </p>
      </div>
    </motion.div>
  );
}
