import React from 'react';
import { CheckCircle2, Download, RefreshCw, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface TranscriptionResultProps {
  text: string;
  onNewTranscription: () => void;
}

export function TranscriptionResult({ text, onNewTranscription }: TranscriptionResultProps) {
  const [copied, setCopied] = React.useState(false);
  const { toast } = useToast();

  const handleDownload = () => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Downloaded!",
      description: "Transcript saved to your device",
    });
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast({
      title: "Copied!",
      description: "Transcript copied to clipboard",
    });
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-slide-up">
      <header className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-success/20 flex items-center justify-center">
          <CheckCircle2 className="w-5 h-5 text-success" />
        </div>
        <div>
          <h2 className="font-display font-semibold text-foreground">Transcription Complete</h2>
          <p className="text-sm text-muted-foreground">Your audio has been transcribed</p>
        </div>
      </header>

      <article className="p-6 rounded-xl bg-card border border-border max-h-96 overflow-y-auto">
        <p className="text-foreground leading-relaxed whitespace-pre-wrap">
          {text}
        </p>
      </article>

      <nav className="flex flex-col sm:flex-row gap-3">
        <Button
          onClick={handleCopy}
          variant="outline"
          className="flex-1 h-11"
        >
          {copied ? (
            <>
              <Check className="w-4 h-4 mr-2" />
              Copied
            </>
          ) : (
            <>
              <Copy className="w-4 h-4 mr-2" />
              Copy Text
            </>
          )}
        </Button>
        <Button
          onClick={handleDownload}
          variant="outline"
          className="flex-1 h-11"
        >
          <Download className="w-4 h-4 mr-2" />
          Download Transcript
        </Button>
        <Button
          onClick={onNewTranscription}
          className="flex-1 h-11 gradient-primary text-primary-foreground"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          New Transcription
        </Button>
      </nav>
    </div>
  );
}
