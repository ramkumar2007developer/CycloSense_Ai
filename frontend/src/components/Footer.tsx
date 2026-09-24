import { Wind } from 'lucide-react';

const FOOTER_LINKS = ['About', 'Features', 'How It Works', 'Contact'];

export default function Footer() {
  const handleScrollLink = (id: string) => {
    const el = document.getElementById(id.toLowerCase().replace(' ', '-'));
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <footer
      id="contact"
      className="relative z-10 border-t"
      style={{ borderColor: 'rgba(255,255,255,0.06)' }}
    >
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-8">
          {/* Brand */}
          <div>
            <div className="mb-2">
              <img 
                src="/cyclonexis-logo.png" 
                alt="CycloNexis AI Logo" 
                className="h-[92px] w-auto object-contain"
              />
            </div>
            <p className="text-xs text-gray-600 mt-1 ml-0.5">
              AI-powered tropical cyclone intelligence.
            </p>
          </div>


        </div>

        {/* Bottom bar */}
        <div
          className="mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center gap-3"
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
        >
          <p className="text-[10px] tracking-widest uppercase text-gray-500">© 2026 CycloNexis AI. ALL RIGHTS RESERVED.</p>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" style={{ animation: 'blink 2s infinite' }} />
            <span className="text-[10px] tracking-widest text-cyan-400/80 font-mono uppercase">AI SYSTEM OPERATIONAL</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
