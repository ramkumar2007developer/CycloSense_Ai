import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Image, FileText, Scan } from 'lucide-react';
import { formatFileSize } from '@/lib/utils';

interface ImageUploaderProps {
  onImageChange: (file: File | null) => void;
  error?: string;
}

export default function ImageUploader({ onImageChange, error }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{ w: number; h: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.match(/image\/(jpeg|jpg|png)/)) {
      alert('Only JPG and PNG files are supported.');
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      alert('File size must be under 10 MB.');
      return;
    }
    const url = URL.createObjectURL(f);
    const img = new window.Image();
    img.onload = () => {
      setImgDimensions({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = url;
    setPreview(url);
    setFile(f);
    onImageChange(f);
  }, [onImageChange]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile]
  );

  const handleRemove = () => {
    setPreview(null);
    setFile(null);
    setImgDimensions(null);
    onImageChange(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Image size={14} className="text-cyan-400" strokeWidth={1.5} />
        <span className="text-sm font-semibold text-white tracking-wide">Satellite Imagery</span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        Upload a JPG or PNG satellite image for cyclone analysis. Max 10 MB.
      </p>

      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className={`drop-zone relative flex flex-col items-center justify-center min-h-[200px] cursor-pointer select-none ${isDragging ? 'dragging' : ''} ${error ? 'border-red-500/40' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            role="button"
            aria-label="Upload satellite image"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              aria-label="Select satellite image file"
            />
            <div
              className="w-14 h-14 rounded-full flex items-center justify-center mb-4"
              style={{
                background: isDragging ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${isDragging ? 'rgba(34,211,238,0.4)' : 'rgba(255,255,255,0.1)'}`,
                transition: 'all 0.2s',
              }}
            >
              <Upload size={22} className={isDragging ? 'text-cyan-400' : 'text-gray-500'} strokeWidth={1.5} />
            </div>
            <p className="text-sm font-medium text-gray-300 mb-1">
              {isDragging ? 'Release to upload' : 'Drop satellite image here'}
            </p>
            <p className="text-xs text-gray-600 mb-3">or click to browse</p>
            <div className="flex items-center gap-3 text-xs text-gray-600">
              <span className="px-2 py-0.5 rounded border border-white/08 bg-white/03">.jpg</span>
              <span className="px-2 py-0.5 rounded border border-white/08 bg-white/03">.jpeg</span>
              <span className="px-2 py-0.5 rounded border border-white/08 bg-white/03">.png</span>
              <span className="text-gray-700">· Max 10MB</span>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.2 }}
            className="relative rounded-xl overflow-hidden"
            style={{
              border: '1px solid rgba(34,211,238,0.2)',
              background: 'rgba(0,0,0,0.4)',
            }}
          >
            {/* Image preview */}
            <div className="relative">
              <img
                src={preview}
                alt="Satellite image preview"
                className="w-full max-h-64 object-cover"
                style={{ filter: 'brightness(0.85)' }}
              />

              {/* Scan animation overlay */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div
                  className="absolute w-full h-0.5"
                  style={{
                    background: 'linear-gradient(90deg, transparent, rgba(34,211,238,0.6), transparent)',
                    animation: 'scanVertical 3s linear infinite',
                  }}
                />
                {/* Corner brackets */}
                {['top-2 left-2', 'top-2 right-2', 'bottom-2 left-2', 'bottom-2 right-2'].map((pos, i) => (
                  <div
                    key={i}
                    className={`absolute ${pos} w-4 h-4`}
                    style={{
                      borderTop: i < 2 ? '1.5px solid rgba(34,211,238,0.6)' : 'none',
                      borderBottom: i >= 2 ? '1.5px solid rgba(34,211,238,0.6)' : 'none',
                      borderLeft: i % 2 === 0 ? '1.5px solid rgba(34,211,238,0.6)' : 'none',
                      borderRight: i % 2 === 1 ? '1.5px solid rgba(34,211,238,0.6)' : 'none',
                    }}
                  />
                ))}
                {/* Center crosshair */}
                <div className="absolute inset-0 flex items-center justify-center opacity-30">
                  <div className="w-8 h-px bg-cyan-400" />
                  <div className="absolute w-px h-8 bg-cyan-400" />
                  <div className="absolute w-3 h-3 rounded-full border border-cyan-400" />
                </div>
              </div>

              {/* Remove button */}
              <button
                onClick={handleRemove}
                className="absolute top-3 right-3 w-7 h-7 rounded-full flex items-center justify-center transition-all hover:scale-110"
                style={{
                  background: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.15)',
                }}
                aria-label="Remove image"
              >
                <X size={13} className="text-gray-300" />
              </button>
            </div>

            {/* File metadata */}
            <div className="px-4 py-3 flex items-center gap-4">
              <FileText size={14} className="text-cyan-400/60 shrink-0" strokeWidth={1.5} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">{file?.name}</p>
                <div className="flex items-center gap-3 mt-0.5">
                  <span className="text-xs text-gray-500">{formatFileSize(file?.size ?? 0)}</span>
                  {imgDimensions && (
                    <>
                      <span className="text-gray-700">·</span>
                      <span className="text-xs text-gray-500">
                        {imgDimensions.w} × {imgDimensions.h}px
                      </span>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-cyan-400">
                <Scan size={12} />
                <span className="font-medium">Ready</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <p className="text-xs text-red-400 mt-2" role="alert">{error}</p>
      )}
    </div>
  );
}
