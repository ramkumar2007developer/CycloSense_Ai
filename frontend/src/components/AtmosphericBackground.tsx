import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  life: number;
  maxLife: number;
}

export default function AtmosphericBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    const particles: Particle[] = [];
    let width = window.innerWidth;
    let height = window.innerHeight;

    canvas.width = width;
    canvas.height = height;

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function createParticle(): Particle {
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.15, // Slightly faster
        vy: (Math.random() - 0.5) * 0.15,
        size: Math.random() * 2 + 1, // Larger particles
        opacity: Math.random() * 0.5 + 0.15, // Higher opacity
        life: 0,
        maxLife: Math.random() * 400 + 200,
      };
    }

    // Init particles (increased count)
    const particleCount = prefersReducedMotion ? 0 : 120;
    for (let i = 0; i < particleCount; i++) {
      particles.push(createParticle());
    }

    function drawAtmosphere() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, width, height);

      if (particles.length === 0) return;

      // Particles
      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;
        p.life++;

        if (p.life > p.maxLife || p.x < 0 || p.x > width || p.y < 0 || p.y > height) {
          particles[i] = createParticle();
          return;
        }

        const lifeRatio = p.life / p.maxLife;
        const fadeIn = lifeRatio < 0.1 ? lifeRatio / 0.1 : 1;
        const fadeOut = lifeRatio > 0.8 ? (1 - lifeRatio) / 0.2 : 1;
        const alpha = p.opacity * fadeIn * fadeOut;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(245, 247, 250, ${alpha})`;
        ctx.fill();
      });
    }

    function animate() {
      drawAtmosphere();
      animationId = requestAnimationFrame(animate);
    }

    animate();

    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* Cinematic Cyclone Background Image */}
      <motion.div
        className="absolute inset-0"
        style={{
          y: scrollY * 0.2, // Parallax effect
          opacity: Math.max(0, 1 - scrollY / 1500) // Fade out on scroll
        }}
      >
        <div 
          className="absolute inset-0 w-full h-full bg-cover bg-center"
          style={{ 
            backgroundImage: 'url(/cyclone-bg-cinematic.jpg)',
            animation: 'cyclone-scale 40s ease-in-out infinite, slow-spin 300s linear infinite',
            transformOrigin: '70% 50%',
            filter: 'brightness(0.8) contrast(1.1) saturate(0.9)' // Increased brightness and saturation
          }}
        />
        
        {/* Dark Cinematic Overlays - reduced opacity for better visibility */}
        <div className="absolute inset-0" style={{ background: 'linear-gradient(to right, rgba(5,6,7,0.85) 0%, rgba(5,6,7,0.3) 40%, rgba(5,6,7,0.05) 100%)' }} />
        <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse at center, transparent 0%, rgba(5,6,7,0.7) 100%)' }} />
      </motion.div>

      {/* Canvas for particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ opacity: 0.9 }} // Increased canvas opacity
      />

      {/* Grid overlay */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px)',
          backgroundSize: '80px 80px',
        }}
      />

      {/* Subtle noise texture */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E")`,
          opacity: 0.3,
        }}
      />
    </div>
  );
}
