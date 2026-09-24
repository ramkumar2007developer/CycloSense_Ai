import { motion } from 'framer-motion';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';

interface SignalBreakdownProps {
  visualSignal: number;
  environmentalSignal: number;
  fusionSignal: number;
}

const SIGNAL_COLORS = {
  visual: '#22d3ee',
  environmental: '#3b82f6',
  fusion: '#8b5cf6',
};

function SignalBar({
  label,
  value,
  color,
  delay = 0,
}: {
  label: string;
  value: number;
  color: string;
  delay?: number;
}) {
  const displayVal = typeof value === 'number' && !isNaN(value) ? value : 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: color, boxShadow: `0 0 6px ${color}80` }}
          />
          <span className="text-xs font-medium text-gray-300">{label}</span>
        </div>
        <span className="text-xs font-mono font-bold" style={{ color }}>
          {(displayVal * 100).toFixed(1)}%
        </span>
      </div>
      <div className="signal-bar-track">
        <motion.div
          className="signal-bar-fill"
          style={{ background: `linear-gradient(90deg, ${color}aa, ${color})` }}
          initial={{ width: '0%' }}
          animate={{ width: `${displayVal * 100}%` }}
          transition={{ duration: 1.2, delay: delay + 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
        />
      </div>
    </motion.div>
  );
}

export default function SignalBreakdown({
  visualSignal,
  environmentalSignal,
  fusionSignal,
}: SignalBreakdownProps) {
  const radarData = [
    {
      signal: 'Visual',
      value: typeof visualSignal === 'number' ? visualSignal * 100 : 0,
    },
    {
      signal: 'Environmental',
      value: typeof environmentalSignal === 'number' ? environmentalSignal * 100 : 0,
    },
    {
      signal: 'Fusion',
      value: typeof fusionSignal === 'number' ? fusionSignal * 100 : 0,
    },
  ];

  return (
    <div>
      <div className="label-xs text-gray-500 mb-4">MULTIMODAL SIGNAL BREAKDOWN</div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
        {/* Radar chart */}
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid
                gridType="polygon"
                stroke="rgba(255,255,255,0.06)"
                radialLines={false}
              />
              <PolarAngleAxis
                dataKey="signal"
                tick={{ fill: '#475569', fontSize: 11, fontFamily: 'Inter' }}
              />
              <Radar
                name="Signal"
                dataKey="value"
                stroke="#22d3ee"
                fill="rgba(34, 211, 238, 0.12)"
                strokeWidth={1.5}
                dot={{ fill: '#22d3ee', r: 3 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Signal bars */}
        <div className="flex flex-col gap-5">
          <SignalBar
            label="Visual Signal"
            value={visualSignal}
            color={SIGNAL_COLORS.visual}
            delay={0}
          />
          <SignalBar
            label="Environmental Signal"
            value={environmentalSignal}
            color={SIGNAL_COLORS.environmental}
            delay={0.1}
          />
          <SignalBar
            label="Fusion Signal"
            value={fusionSignal}
            color={SIGNAL_COLORS.fusion}
            delay={0.2}
          />
        </div>
      </div>
    </div>
  );
}
