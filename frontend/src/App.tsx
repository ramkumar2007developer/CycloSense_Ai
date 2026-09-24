import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import AtmosphericBackground from '@/components/AtmosphericBackground';
import Home from '@/pages/Home';
import Analyze from '@/pages/Analyze';

export default function App() {
  return (
    <BrowserRouter>
      <div className="atmospheric-bg noise-overlay min-h-screen flex flex-col">
        <AtmosphericBackground />
        <Navbar />
        <div className="flex-1 relative z-10">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
