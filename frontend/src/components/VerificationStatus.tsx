import { motion } from 'framer-motion';
import { ShieldCheck, ShieldOff, AlertTriangle } from 'lucide-react';
import { VERIFICATION_MESSAGES } from '@/types/cyclone';

interface VerificationStatusProps {
  status: string;
  message?: string;
}

export default function VerificationStatus({ status, message }: VerificationStatusProps) {
  const info = VERIFICATION_MESSAGES[status] ?? {
    title: status.replace(/_/g, ' '),
    description: message ?? 'Verification status received from backend.',
    color: '#94a3b8',
  };

  const IconMap: Record<string, React.ElementType> = {
    VERIFIED_CYCLONE: ShieldCheck,
    LLM_NOT_REQUIRED: ShieldOff,
    HUMAN_REVIEW_REQUIRED: AlertTriangle,
  };

  const Icon = IconMap[status] ?? ShieldCheck;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <div className="label-xs text-gray-500 mb-4">AI VERIFICATION</div>

      <div
        className="rounded-xl p-5 relative overflow-hidden"
        style={{
          background: `${info.color}08`,
          border: `1px solid ${info.color}25`,
        }}
      >
        {/* Background glow */}
        <div
          className="absolute top-0 right-0 w-32 h-32 pointer-events-none"
          style={{
            background: `radial-gradient(ellipse at top right, ${info.color}12, transparent)`,
          }}
        />

        <div className="relative flex items-start gap-4">
          {/* Icon */}
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
            style={{
              background: `${info.color}15`,
              border: `1px solid ${info.color}30`,
            }}
          >
            <Icon size={18} style={{ color: info.color }} strokeWidth={1.5} />
          </div>

          <div className="flex-1">
            {/* Status badge */}
            <div className="flex items-center gap-2 mb-2">
              <span
                className="inline-flex items-center gap-1.5 text-xs font-bold tracking-widest uppercase px-2.5 py-1 rounded-full"
                style={{
                  background: `${info.color}15`,
                  color: info.color,
                  border: `1px solid ${info.color}30`,
                }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    background: info.color,
                    boxShadow: `0 0 5px ${info.color}`,
                    animation: 'blink 2s ease-in-out infinite',
                  }}
                />
                {info.title}
              </span>
            </div>

            {/* Raw status */}
            <div className="text-xs font-mono text-gray-600 mb-2">{status}</div>

            {/* Description */}
            <p className="text-sm text-gray-400 leading-relaxed">
              {info.description}
            </p>

            {/* Backend message if provided */}
            {message && message !== info.description && (
              <p className="text-xs text-gray-600 mt-2 leading-relaxed italic">
                {message}
              </p>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
