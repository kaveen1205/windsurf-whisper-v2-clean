import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/FileUpload';
import { TranscriptionResult } from '@/components/TranscriptionResult';
import { Mic, LogOut, Youtube } from 'lucide-react';

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleFileSelect = async (file: File) => {
    try {
      console.log('[Dashboard] Starting transcription for file:', file.name);
      setFile(file);
      setError(null);
      setLoading(true);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', 'auto');
      // Highest accuracy: use default server settings (no fast mode)

      console.log('[Dashboard] Sending request to http://localhost:8081/transcribe-start');
      const startRes = await fetch('http://localhost:8081/transcribe-start', {
        method: 'POST',
        body: formData,
      });

      if (!startRes.ok) {
        const raw = await startRes.text().catch(() => '');
        let message = 'Failed to start transcription.';
        if (raw) {
          try {
            const parsed = JSON.parse(raw);
            message = parsed?.detail || parsed?.error || JSON.stringify(parsed);
          } catch (_e) {
            message = raw;
          }
        }
        console.error('[Dashboard] /transcribe-start error:', raw);
        setError(message);
        return;
      }

      const { job_id } = await startRes.json();
      if (!job_id) {
        throw new Error('No job_id returned by server.');
      }

      // Poll progress
      setProgress(5);
      const pollIntervalMs = 700;
      await new Promise<void>((resolve, reject) => {
        const id = setInterval(async () => {
          try {
            const progRes = await fetch(`http://localhost:8081/transcribe-progress?job_id=${job_id}`);
            if (!progRes.ok) {
              const raw = await progRes.text().catch(() => '');
              throw new Error(raw || 'Progress polling failed');
            }
            const prog = await progRes.json();
            const pct = typeof prog.percent === 'number' ? Math.max(0, Math.min(100, prog.percent)) : 0;
            setProgress(pct);
            if (prog.status === 'done' || pct >= 100) {
              clearInterval(id);
              const resRes = await fetch(`http://localhost:8081/transcribe-result?job_id=${job_id}`);
              if (!resRes.ok) {
                const raw = await resRes.text().catch(() => '');
                throw new Error(raw || 'Failed to fetch result');
              }
              const data = await resRes.json().catch(() => null);
              if (!data || typeof data.text !== 'string') {
                throw new Error('Unexpected result format from transcription service.');
              }
              setResult(data.text);
              setProgress(100);
              resolve();
            } else if (prog.status === 'error') {
              clearInterval(id);
              reject(new Error(prog.message || 'Transcription failed'));
            }
          } catch (e: any) {
            clearInterval(id);
            reject(e);
          }
        }, pollIntervalMs);
      });
    } catch (err: any) {
      console.error('Transcription error:', err);
      setError(err?.message || 'Something went wrong during transcription.');
    } finally {
      setLoading(false);
    }
  };

  const handleNewTranscription = () => {
    setFile(null);
    setResult('');
    setError(null);
    setProgress(0);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container max-w-4xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl gradient-primary flex items-center justify-center">
              <Mic className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-display font-semibold text-foreground">Panuval Maatram</span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/youtube')}
              className="text-muted-foreground hover:text-foreground"
            >
              <Youtube className="w-4 h-4 mr-2" />
              YouTube Live
            </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLogout}
            className="text-muted-foreground hover:text-foreground"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Sign out
          </Button>
          </div>
        </div>
      </header>

      <main className="container max-w-2xl mx-auto px-4 py-8 md:py-12 space-y-10">
        <section className="text-center animate-slide-up">
          <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-2">
            Welcome, {user?.username || 'User'}
          </h1>
          <p className="text-muted-foreground mb-8">
            Transform your audio into text with AI-powered transcription
          </p>
          <article className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
            {[
              { title: 'Fast', desc: 'Get results in seconds' },
              { title: 'Accurate', desc: 'Powered by advanced AI' },
              { title: 'Simple', desc: 'Just upload and transcribe' },
            ].map((feature) => (
              <div key={feature.title} className="p-4 rounded-xl bg-card border border-border">
                <h3 className="font-display font-semibold text-foreground mb-1">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.desc}</p>
              </div>
            ))}
          </article>
        </section>

        <section className="animate-slide-up space-y-6">
          <header className="text-center">
            <h2 className="text-2xl font-display font-bold text-foreground mb-2">
              Upload your file
            </h2>
            <p className="text-muted-foreground">
              Select an audio or video file to transcribe
            </p>
          </header>

          <FileUpload
            onFileSelect={handleFileSelect}
            isUploading={loading}
            uploadProgress={progress}
          />

          {error && (
            <p className="text-sm text-destructive text-center">{error}</p>
          )}
        </section>

        {result && (
          <section className="animate-slide-up">
            <TranscriptionResult
              text={result}
              onNewTranscription={handleNewTranscription}
            />
          </section>
        )}
      </main>
    </div>
  );
}
