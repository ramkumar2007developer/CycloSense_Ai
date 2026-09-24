import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, Circle, Satellite, AlertTriangle } from 'lucide-react';
import { useEffect, useRef } from 'react';

function CycloneHUD() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;
    let animId: number;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);

      // Radar rings
      const rings = [60, 110, 165, 225];
      rings.forEach((r, i) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(34, 211, 238, ${0.12 - i * 0.02})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Radar sweep
      const sweepAngle = ((frame * 1.5) * Math.PI) / 180;
      // Sweep line
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(
        cx + Math.cos(sweepAngle) * 225,
        cy + Math.sin(sweepAngle) * 225
      );
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Sweep trail
      for (let t = 0; t < 60; t++) {
        const angle = sweepAngle - (t * Math.PI) / 120;
        const alpha = (1 - t / 60) * 0.12;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, 225, angle, angle + Math.PI / 120);
        ctx.closePath();
        ctx.fillStyle = `rgba(34, 211, 238, ${alpha})`;
        ctx.fill();
      }

      // Center dot
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(34, 211, 238, 0.9)';
      ctx.fill();

      // Pulsing ring
      const pulseR = 20 + Math.sin(frame * 0.05) * 8;
      ctx.beginPath();
      ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.4 + Math.sin(frame * 0.05) * 0.2})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Crosshair
      const chSize = 14;
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
      ctx.lineWidth = 1;
      // Top
      ctx.beginPath(); ctx.moveTo(cx, cy - 225); ctx.lineTo(cx, cy - 225 + chSize); ctx.stroke();
      // Bottom
      ctx.beginPath(); ctx.moveTo(cx, cy + 225); ctx.lineTo(cx, cy + 225 - chSize); ctx.stroke();
      // Left
      ctx.beginPath(); ctx.moveTo(cx - 225, cy); ctx.lineTo(cx - 225 + chSize, cy); ctx.stroke();
      // Right
      ctx.beginPath(); ctx.moveTo(cx + 225, cy); ctx.lineTo(cx + 225 - chSize, cy); ctx.stroke();

      // Tick marks
      for (let i = 0; i < 24; i++) {
        const angle = (i * Math.PI * 2) / 24;
        const isLong = i % 6 === 0;
        const r1 = 225 - (isLong ? 12 : 6);
        const r2 = 225;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
        ctx.lineTo(cx + Math.cos(angle) * r2, cy + Math.sin(angle) * r2);
        ctx.strokeStyle = `rgba(34, 211, 238, ${isLong ? 0.4 : 0.2})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Blip - cyclone detection dot
      const blipAlpha = 0.5 + Math.sin(frame * 0.08) * 0.5;
      ctx.beginPath();
      ctx.arc(cx + 40, cy - 35, 5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(239, 68, 68, ${blipAlpha})`;
      ctx.fill();
      // Blip glow
      ctx.beginPath();
      ctx.arc(cx + 40, cy - 35, 12, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(239, 68, 68, ${blipAlpha * 0.15})`;
      ctx.fill();

      frame++;
      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={470}
      height={470}
      className="absolute inset-0 m-auto"
      style={{ maxWidth: '100%', maxHeight: '100%' }}
    />
  );
}

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden pt-16">
      {/* Background deep atmospheric radial */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 65% 50%, rgba(8,24,44,0.5) 0%, transparent 60%)',
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-6 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-8 items-center min-h-[calc(100vh-4rem)] py-16">

          {/* ─── Left: Text Content ─── */}
          <div className="flex flex-col gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <span
                className="inline-flex items-center gap-2 text-xs font-semibold tracking-[0.2em] uppercase"
                style={{ color: '#22d3ee' }}
              >
                <span className="w-4 h-px bg-cyan-400" />
                Smarter Insights. Safer Tomorrows.
              </span>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
            >
              <h1 className="text-6xl sm:text-7xl lg:text-8xl font-black tracking-tight leading-[0.92]">
                <span className="text-white">Cyclo</span>
                <span
                  style={{
                    background: 'linear-gradient(135deg, #22d3ee, #3b82f6)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}
                >
                  Sense
                </span>
                <br />
                <span className="text-white">AI</span>
              </h1>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.35 }}
            >
              <p className="text-xl font-light text-gray-300 leading-tight">
                AI-Powered Tropical Cyclone<br />
                Prediction & Verification
              </p>
            </motion.div>

            <motion.p
              className="text-sm text-gray-500 leading-relaxed max-w-md"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.45 }}
            >
              Combining satellite imagery, environmental data and advanced AI/ML models with 
              LLM verification to detect and analyze tropical cyclones with greater accuracy 
              and explainability.
            </motion.p>

            <motion.div
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.55 }}
            >
              <Link to="/analyze" className="btn-primary text-sm">
                <ArrowRight size={16} />
                Get Started
              </Link>
              <button
                onClick={() => {
                  document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
                }}
                className="btn-ghost text-sm"
                aria-label="Learn more about CycloNexis AI"
              >
                <Circle size={14} strokeWidth={2} className="text-cyan-400/60" />
                Learn More
              </button>
            </motion.div>

            {/* Status strip */}
            <motion.div
              className="flex items-center gap-6 pt-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <div className="flex items-center gap-2">
                <span className="status-dot active" />
                <span className="text-xs text-gray-500 font-medium">System Online</span>
              </div>
              <div className="w-px h-4 bg-white/10" />
              <div className="text-xs text-gray-600 font-mono">v2.1.0 · PROD</div>
              <div className="w-px h-4 bg-white/10" />
              <div className="text-xs text-gray-600">12 Parameters · 3 Signals</div>
            </motion.div>
          </div>

          {/* ─── Right: Cyclone HUD ─── */}
          <motion.div
            className="relative flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9, delay: 0.3 }}
          >
            {/* Outer glow */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: '500px',
                height: '500px',
                background: 'radial-gradient(ellipse, rgba(8, 28, 56, 0.6) 0%, transparent 70%)',
              }}
            />

            {/* Main radar container */}
            <div
              className="relative rounded-full overflow-hidden"
              style={{
                width: 'min(470px, 90vw)',
                height: 'min(470px, 90vw)',
                background: 'radial-gradient(ellipse at center, rgba(4,16,32,0.9) 0%, rgba(2,8,16,0.98) 100%)',
                border: '1px solid rgba(34, 211, 238, 0.15)',
                boxShadow: '0 0 60px rgba(34, 211, 238, 0.08), inset 0 0 40px rgba(0,0,0,0.5)',
              }}
            >
              <CycloneHUD />

              {/* Cyclone image overlay */}
              <div
                className="absolute inset-0 rounded-full overflow-hidden"
                style={{
                  background: 'url(/cyclone-bg.jpg) center/cover',
                  opacity: 0.12,
                  mixBlendMode: 'screen',
                }}
              />
            </div>

            {/* HUD Cards */}
            {/* Top-left card */}
            <motion.div
              className="absolute top-4 left-0 glass-panel rounded-xl px-4 py-3"
              style={{ minWidth: '180px' }}
              animate={{ y: [0, -4, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <AlertTriangle size={12} className="text-red-400" />
                <span className="text-xs font-bold tracking-widest uppercase text-red-400">
                  CYCLONE DETECTED
                </span>
              </div>
              <div className="text-white font-bold text-base">High Risk</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 bg-white/05 rounded-full h-1.5">
                  <div className="h-full rounded-full bg-gradient-to-r from-red-500 to-orange-400" style={{ width: '77.5%' }} />
                </div>
                <span className="text-xs font-mono text-gray-400">77.5</span>
              </div>
            </motion.div>

            {/* Bottom-right card */}
            <motion.div
              className="absolute bottom-8 right-0 glass-panel rounded-xl px-4 py-3"
              animate={{ y: [0, 4, 0] }}
              transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="w-2 h-2 rounded-full bg-cyan-400"
                  style={{ animation: 'blink 1.5s ease-in-out infinite' }}
                />
                <span className="text-xs font-semibold tracking-widest uppercase text-cyan-400">
                  ANALYZING...
                </span>
              </div>
              <div className="text-xs text-gray-500 mt-1 font-mono">
                Visual · ENV · Fusion
              </div>
            </motion.div>

            {/* Coordinates overlay */}
            <div className="absolute top-8 right-4 hud-label text-right">
              <div>14.2°N 86.7°E</div>
              <div className="mt-0.5 text-gray-600">BAY OF BENGAL</div>
            </div>

            {/* Satellite label */}
            <div className="absolute bottom-6 left-2 flex items-center gap-1.5">
              <Satellite size={10} className="text-cyan-400/50" />
              <span className="hud-label text-xs" style={{ fontSize: '9px' }}>GOES-16 · IR CH13</span>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div
        className="absolute bottom-0 left-0 right-0 h-32 pointer-events-none"
        style={{ background: 'linear-gradient(to bottom, transparent, rgba(5,7,9,0.8))' }}
      />
    </section>
  );
}
