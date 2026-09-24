import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Wind, Menu, X, ArrowRight, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PROJECT_INFO_URL = 'https://cyclone-info-site.vercel.app/';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [exploreHover, setExploreHover] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <header
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-500"
        style={{
          background: scrolled
            ? 'rgba(5, 7, 10, 0.75)'
            : 'transparent',
          backdropFilter: scrolled ? 'blur(20px)' : 'blur(0px)',
          WebkitBackdropFilter: scrolled ? 'blur(20px)' : 'blur(0px)',
          borderBottom: scrolled ? '1px solid rgba(255,255,255,0.06)' : '1px solid transparent',
        }}
      >
        <nav className="max-w-7xl mx-auto px-6 h-24 flex items-center justify-between" aria-label="Main navigation">

          {/* LEFT — Brand */}
          <Link to="/" className="flex items-center group transition-transform duration-300 hover:scale-105" aria-label="CycloNexis AI Home">
            <img 
              src="/cyclonexis-logo.png" 
              alt="CycloNexis AI Logo" 
              className="h-[76px] w-auto object-contain"
            />
          </Link>

          {/* RIGHT — Desktop Actions */}
          <div className="hidden md:flex items-center gap-3">

          {/* EXPLORE PROJECT DETAILS */}
          {PROJECT_INFO_URL ? (
            <a
              href={PROJECT_INFO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 transition-all duration-250"
              style={{
                padding: '9px 18px',
                borderRadius: '10px',
                border: exploreHover
                  ? '1px solid rgba(255,255,255,0.28)'
                  : '1px solid rgba(255,255,255,0.16)',
                background: exploreHover
                  ? 'rgba(255,255,255,0.06)'
                  : 'rgba(255,255,255,0.02)',
                color: exploreHover ? '#22d3ee' : '#F5F7FA',
                fontSize: '11px',
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase' as const,
                boxShadow: exploreHover
                  ? '0 0 18px rgba(34,211,238,0.08)'
                  : 'none',
                transition: 'all 0.25s ease',
              }}
              onMouseEnter={() => setExploreHover(true)}
              onMouseLeave={() => setExploreHover(false)}
            >
              EXPLORE PROJECT DETAILS
              <motion.span
                animate={{ x: exploreHover ? 4 : 0 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
              >
                <ArrowRight size={13} />
              </motion.span>
            </a>
          ) : (
            // Coming soon state — URL not yet configured
            <div
              className="flex items-center gap-2"
              title="Project info website coming soon"
              style={{
                padding: '9px 18px',
                borderRadius: '10px',
                border: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(255,255,255,0.01)',
                color: 'rgba(255,255,255,0.3)',
                fontSize: '11px',
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase' as const,
                cursor: 'default',
              }}
            >
              <Clock size={11} className="opacity-50" />
              EXPLORE PROJECT DETAILS
              <span
                style={{
                  fontSize: '8px',
                  letterSpacing: '0.1em',
                  color: '#22d3ee',
                  opacity: 0.6,
                  fontWeight: 700,
                }}
              >
                SOON
              </span>
            </div>
          )}

          {/* UIverse Explore Button */}
          <button
            type="button"
            onClick={() => PROJECT_INFO_URL && window.open(PROJECT_INFO_URL, '_blank')}
            className="flex justify-center gap-1.5 items-center shadow-xl text-sm bg-gray-50 backdrop-blur-md font-semibold isolation-auto border-gray-50 before:absolute before:w-full before:transition-all before:duration-700 before:hover:w-full before:-left-full before:hover:left-0 before:rounded-full before:bg-emerald-500 hover:text-white before:-z-10 before:aspect-square before:hover:scale-150 before:hover:duration-700 relative z-10 px-3 py-1.5 overflow-hidden border-2 rounded-full group text-gray-900"
            title={PROJECT_INFO_URL ? 'Explore project details' : 'Coming soon'}
          >
            Explore
            <svg
              className="w-6 h-6 justify-end group-hover:rotate-90 group-hover:bg-gray-50 ease-linear duration-300 rounded-full border border-gray-700 group-hover:border-none p-1.5 rotate-45"
              viewBox="0 0 16 19"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M7 18C7 18.5523 7.44772 19 8 19C8.55228 19 9 18.5523 9 18H7ZM8.70711 0.292893C8.31658 -0.0976311 7.68342 -0.0976311 7.29289 0.292893L0.928932 6.65685C0.538408 7.04738 0.538408 7.68054 0.928932 8.07107C1.31946 8.46159 1.95262 8.46159 2.34315 8.07107L8 2.41421L13.6569 8.07107C14.0474 8.46159 14.6805 8.46159 15.0711 8.07107C15.4616 7.68054 15.4616 7.04738 15.0711 6.65685L8.70711 0.292893ZM9 18L9 1H7L7 18H9Z"
                className="fill-gray-800 group-hover:fill-gray-800"
              />
            </svg>
          </button>

            {/* CYCLONE PREDICTION SYSTEM */}
            <Link
              to="/analyze"
              className="text-[11px] font-bold tracking-[0.1em] uppercase px-5 py-2.5 rounded-lg transition-all duration-300"
              style={{
                border: '1px solid rgba(255,255,255,0.12)',
                color: 'rgba(255,255,255,0.9)',
                background: 'rgba(255,255,255,0.04)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(34,211,238,0.5)';
                e.currentTarget.style.color = '#fff';
                e.currentTarget.style.background = 'rgba(34,211,238,0.07)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)';
                e.currentTarget.style.color = 'rgba(255,255,255,0.9)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
              }}
            >
              Cyclone Prediction System
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 text-gray-400 hover:text-white transition-colors"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </nav>
      </header>

      {/* Mobile nav drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="fixed top-24 left-0 right-0 z-40 md:hidden"
            style={{
              background: 'rgba(5, 7, 10, 0.97)',
              backdropFilter: 'blur(24px)',
              borderBottom: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <nav className="px-6 py-6 flex flex-col gap-4" aria-label="Mobile navigation">

              {/* EXPLORE PROJECT DETAILS */}
              {PROJECT_INFO_URL ? (
                <a
                  href={PROJECT_INFO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between gap-2 px-5 py-4 rounded-xl"
                  style={{
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: 'rgba(255,255,255,0.02)',
                    color: '#F5F7FA',
                    fontSize: '12px',
                    fontWeight: 600,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase' as const,
                  }}
                  onClick={() => setMobileOpen(false)}
                >
                  <span>EXPLORE PROJECT DETAILS</span>
                  <ArrowRight size={14} className="text-cyan-400/70" />
                </a>
              ) : (
                <div
                  className="flex items-center justify-between gap-2 px-5 py-4 rounded-xl"
                  style={{
                    border: '1px solid rgba(255,255,255,0.06)',
                    background: 'rgba(255,255,255,0.01)',
                    color: 'rgba(255,255,255,0.3)',
                    fontSize: '12px',
                    fontWeight: 600,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase' as const,
                    cursor: 'default',
                  }}
                >
                  <span>EXPLORE PROJECT DETAILS</span>
                  <span style={{ fontSize: '9px', color: '#22d3ee', opacity: 0.6, fontWeight: 700, letterSpacing: '0.1em' }}>SOON</span>
                </div>
              )}

              {/* CYCLONE PREDICTION SYSTEM */}
              <Link
                to="/analyze"
                className="flex items-center justify-center px-5 py-4 rounded-xl text-[12px] font-bold tracking-[0.1em] uppercase transition-all"
                style={{
                  border: '1px solid rgba(255,255,255,0.12)',
                  background: 'rgba(255,255,255,0.05)',
                  color: '#fff',
                }}
                onClick={() => setMobileOpen(false)}
              >
                Cyclone Prediction System
              </Link>

            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
