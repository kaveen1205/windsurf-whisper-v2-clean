import React, { useState, useRef } from 'react';
import { Upload, FileAudio, X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isUploading: boolean;
  uploadProgress: number;
}

const ACCEPTED_FORMATS = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav', 'audio/mp4', 'audio/aac', 'audio/x-m4a', 'video/mp4', 'video/quicktime', 'video/x-matroska'];
const ACCEPTED_EXTENSIONS = '.mp3,.wav,.mp4,.m4a,.mov,.mkv';

export function FileUpload({ onFileSelect, isUploading, uploadProgress }: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFile && !isUploading) {
      onFileSelect(selectedFile);
    }
  };

  const validateFile = (file: File): boolean => {
    if (!ACCEPTED_FORMATS.includes(file.type)) {
      setError('Please upload an MP3, WAV, or MP4 file');
      return false;
    }
    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      setError('File size must be less than 100MB');
      return false;
    }
    setError(null);
    return true;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (selectedFile) {
      return;
    }
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        console.log('[FileUpload] File selected via input:', file.name);
        setSelectedFile(file);
      }
    }
  };

  // Prevent double trigger: we rely solely on <form onSubmit> to call onFileSelect

  const clearSelection = () => {
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isLocked = !!selectedFile || isUploading;

  return (
    <form onSubmit={handleSubmit} className="w-full space-y-4">
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ${
          'border-border hover:border-primary/50 hover:bg-accent/30'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLocked}
        />

        <div className="flex flex-col items-center text-center">
          <div className="w-14 h-14 rounded-xl bg-accent flex items-center justify-center mb-4">
            <Upload className="w-7 h-7 text-accent-foreground" />
          </div>
          <h3 className="font-display font-semibold text-foreground mb-1">
            Click to upload
          </h3>
          <p className="text-sm text-muted-foreground mb-2">
            (paste & drag-drop disabled)
          </p>
          <p className="text-xs text-muted-foreground">
            Supports MP3, WAV, MP4, M4A, MOV, MKV (max 100MB)
          </p>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          {error}
        </div>
      )}

      {selectedFile && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-card border border-border">
          <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
            <FileAudio className="w-5 h-5 text-accent-foreground" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-medium text-foreground truncate">{selectedFile.name}</p>
            <p className="text-sm text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
          </div>
          <button
            onClick={clearSelection}
            className="p-2 hover:bg-accent rounded-lg transition-colors"
            aria-label="Remove file"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      )}

      {isUploading && (
        <div className="space-y-3 p-4 rounded-xl bg-card border border-border">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-primary animate-spin" />
            <span className="text-sm font-medium text-foreground">Uploading...</span>
            <span className="ml-auto text-sm text-muted-foreground">{uploadProgress}%</span>
          </div>
          <Progress value={uploadProgress} className="h-2" />
        </div>
      )}

      {selectedFile && (
        <Button
          type="submit"
          disabled={isUploading || !selectedFile}
          className="w-full h-12 gradient-primary text-primary-foreground font-medium flex items-center justify-center gap-2"
        >
          {isUploading && <Loader2 className="w-4 h-4 animate-spin" />}
          {isUploading ? 'Transcribing…' : 'Start Transcription'}
        </Button>
      )}
    </form>
  );
}
