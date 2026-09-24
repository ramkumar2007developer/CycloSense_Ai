import { useEffect, useRef } from 'react';
import { getRiskLevel } from '@/types/cyclone';

interface RiskGaugeProps {
  value: number; // 0-100
}

export default function RiskGauge({ value }: RiskGaugeProps) {
  const riskLevel = getRiskLevel(value);
  const animatedRef = useRef<number>(0);
  const frameRef = useRef<number>(0);
  const displayRef = useRef<HTMLSpanElement>(null);
  const fillRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    const target = value;
    const start = Date.now();
    const duration = 1800;

    const W = 200;
    const H = 200;
    const cx = W / 2;
    const cy = H / 2;
    const R = 80;
    const startAngle = Math.PI * 0.75; // 135°
    const endAngle = Math.PI * 2.25; // 405° (270° arc)

    function getArcPath(val: number) {
      const arcEnd = startAngle + (val / 100) * (endAngle - startAngle);
      const x1 = cx + R * Math.cos(startAngle);
      const y1 = cy + R * Math.sin(startAngle);
      const x2 = cx + R * Math.cos(arcEnd);
      const y2 = cy + R * Math.sin(arcEnd);
      const largeArc = val > 50 ? 1 : 0;
      return `M ${x1} ${y1} A ${R} ${R} 0 ${largeArc} 1 ${x2} ${y2}`;
    }

    function animate() {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * target;
      animatedRef.current = current;

      if (displayRef.current) {
        displayRef.current.textContent = current.toFixed(1);
      }
      if (fillRef.current) {
        fillRef.current.setAttribute('d', getArcPath(current));
      }

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      }
    }

    frameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameRef.current);
  }, [value]);

  const riskColors: Record<string, string> = {
    LOW: '#22d3ee',
    MODERATE: '#f59e0b',
    HIGH: '#ef4444',
  };

  const W = 200;
  const H = 200;
  const cx = W / 2;
  const cy = H / 2;
  const R = 80;
  const startAngle = Math.PI * 0.75;
  const endAngle = Math.PI * 2.25;

  // Track path (full 270°)
  const tx1 = cx + R * Math.cos(startAngle);
  const ty1 = cy + R * Math.sin(startAngle);
  const tx2 = cx + R * Math.cos(endAngle - 0.001);
  const ty2 = cy + R * Math.sin(endAngle - 0.001);
  const trackPath = `M ${tx1} ${ty1} A ${R} ${R} 0 1 1 ${tx2} ${ty2}`;

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="label-xs text-gray-500 text-center">DEVELOPMENT / RISK INDEX</div>

      {/* Gauge SVG */}
      <div className="relative">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
          {/* Track */}
          <path
            d={trackPath}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={10}
            strokeLinecap="round"
          />

          {/* Low zone tick */}
          <path
            d={`M ${cx + R * Math.cos(startAngle + 0.4 * (endAngle - startAngle))} ${cy + R * Math.sin(startAngle + 0.4 * (endAngle - startAngle))} A ${R} ${R} 0 0 1 ${cx + R * Math.cos(startAngle + 0.4 * (endAngle - startAngle) + 0.01)} ${cy + R * Math.sin(startAngle + 0.4 * (endAngle - startAngle) + 0.01)}`}
            fill="none"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth={14}
            strokeLinecap="butt"
          />
          {/* Moderate zone tick */}
          <path
            d={`M ${cx + R * Math.cos(startAngle + 0.65 * (endAngle - startAngle))} ${cy + R * Math.sin(startAngle + 0.65 * (endAngle - startAngle))} A ${R} ${R} 0 0 1 ${cx + R * Math.cos(startAngle + 0.65 * (endAngle - startAngle) + 0.01)} ${cy + R * Math.sin(startAngle + 0.65 * (endAngle - startAngle) + 0.01)}`}
            fill="none"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth={14}
            strokeLinecap="butt"
          />

          {/* Fill arc (animated via ref) */}
          <path
            ref={fillRef}
            d={`M ${tx1} ${ty1} A ${R} ${R} 0 0 1 ${tx1 + 0.1} ${ty1}`}
            fill="none"
            stroke={riskColors[riskLevel]}
            strokeWidth={10}
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 8px ${riskColors[riskLevel]}60)`,
            }}
          />

          {/* Center value */}
          <foreignObject x={cx - 60} y={cy - 30} width={120} height={60}>
            <div className="flex flex-col items-center">
              <div className="flex items-baseline gap-0.5">
                <span
                  ref={displayRef}
                  className="text-3xl font-black text-white font-mono"
                >
                  0.0
                </span>
                <span className="text-sm text-gray-500 font-mono">/100</span>
              </div>
              <span
                className="text-xs font-bold tracking-widest uppercase mt-0.5"
                style={{ color: riskColors[riskLevel] }}
              >
                {riskLevel}
              </span>
            </div>
          </foreignObject>
        </svg>

        {/* Glow effect */}
        <div
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            background: `radial-gradient(ellipse at center, ${riskColors[riskLevel]}08 0%, transparent 70%)`,
          }}
        />
      </div>

      {/* Risk label badges */}
      <div className="flex items-center gap-2">
        {(['LOW', 'MODERATE', 'HIGH'] as const).map((level) => (
          <div
            key={level}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wider"
            style={{
              background: riskLevel === level ? `${riskColors[level]}15` : 'rgba(255,255,255,0.04)',
              border: `1px solid ${riskLevel === level ? riskColors[level] + '40' : 'rgba(255,255,255,0.08)'}`,
              color: riskLevel === level ? riskColors[level] : '#475569',
              transition: 'all 0.3s',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: riskLevel === level ? riskColors[level] : '#475569',
                boxShadow: riskLevel === level ? `0 0 6px ${riskColors[level]}` : 'none',
              }}
            />
            {level}
          </div>
        ))}
      </div>
    </div>
  );
}
