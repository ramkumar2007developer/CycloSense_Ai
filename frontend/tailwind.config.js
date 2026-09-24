/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        // Core backgrounds
        bg: {
          primary: '#050709',
          secondary: '#080c10',
          tertiary: '#0d1117',
        },
        // Glass surfaces
        glass: {
          DEFAULT: 'rgba(255,255,255,0.03)',
          hover: 'rgba(255,255,255,0.06)',
          border: 'rgba(255,255,255,0.08)',
          strong: 'rgba(255,255,255,0.10)',
        },
        // Accents
        accent: {
          cyan: '#22d3ee',
          'cyan-dim': '#0891b2',
          amber: '#f59e0b',
          crimson: '#ef4444',
          blue: '#3b82f6',
          purple: '#8b5cf6',
        },
        // Text hierarchy
        text: {
          primary: '#f8fafc',
          secondary: '#94a3b8',
          muted: '#475569',
          accent: '#22d3ee',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-atmospheric': 'radial-gradient(ellipse at top, #0f1b2d 0%, #050709 60%)',
      },
      animation: {
        'spin-slow': 'spin 20s linear infinite',
        'spin-slow-reverse': 'spin 30s linear infinite reverse',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-scan': 'radar 4s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'scan-line': 'scanLine 3s linear infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'fade-in': 'fadeIn 0.5s ease-out forwards',
      },
      keyframes: {
        radar: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        scanLine: {
          '0%': { top: '0%', opacity: '0' },
          '10%': { opacity: '1' },
          '90%': { opacity: '1' },
          '100%': { top: '100%', opacity: '0' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(34, 211, 238, 0.15)',
        'glow-amber': '0 0 20px rgba(245, 158, 11, 0.15)',
        'glow-red': '0 0 20px rgba(239, 68, 68, 0.15)',
        'glass': '0 4px 24px rgba(0,0,0,0.4)',
        'glass-lg': '0 8px 40px rgba(0,0,0,0.5)',
      },
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
      },
    },
  },
  plugins: [],
};
