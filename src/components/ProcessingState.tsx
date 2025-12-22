import React from 'react';
import { Loader2, AudioWaveform } from 'lucide-react';

export function ProcessingState() {
  return (
    <div className="flex flex-col items-center py-12 animate-slide-up">
      <div className="relative mb-6">
        <div className="w-20 h-20 rounded-2xl bg-accent flex items-center justify-center">
          <AudioWaveform className="w-10 h-10 text-primary animate-pulse-soft" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <Loader2 className="w-4 h-4 text-primary-foreground animate-spin" />
        </div>
      </div>
      
      <h3 className="font-display font-semibold text-xl text-foreground mb-2">
        Processing your audio
      </h3>
      <p className="text-muted-foreground text-center max-w-sm">
        Our AI is transcribing your file. This usually takes a few moments depending on the file length.
      </p>

      <div className="flex gap-1 mt-6">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-primary animate-pulse-soft"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
    </div>
  );
}
