import React, { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Link as LinkIcon, Loader2 } from 'lucide-react';

const API_BASE = (() => {
  try {
    const env: any = (import.meta as any)?.env;
    const fromEnv = env?.VITE_API_BASE || (window as any)?.__API_BASE__;
    if (fromEnv) return String(fromEnv);
    const origin = location.origin;
    // Map 8082 -> 8081 (common dev pairing)
    if (/:\b8082\b/.test(origin)) return origin.replace(/:8082\b/, ':8081');
    // If running on localhost/127.0.0.1 but NOT 8081, assume backend is 8081
    if (/^https?:\/\/(localhost|127\.0\.0\.1)(?::(?!8081)\d+)?\/?$/.test(origin)) {
      return `${location.protocol}//${location.hostname}:8081`;
    }
    return origin;
  } catch {
    return 'http://localhost:8081';
  }
})();

let RESOLVED_API = API_BASE;
const getApiCandidates = (): string[] => {
  const out: string[] = [];
  const push = (v?: string) => {
    if (!v) return;
    if (!out.includes(v)) out.push(v);
  };
  try {
    const env: any = (import.meta as any)?.env;
    const fromEnv = env?.VITE_API_BASE || (window as any)?.__API_BASE__;
    push(fromEnv ? String(fromEnv) : undefined);
  } catch {}
  push(RESOLVED_API);
  try {
    const origin = location.origin;
    push(origin);
    // Common dev mappings: 8082/8080 -> backend 8081 or 8000
    if (/:\b8082\b/.test(origin)) {
      push(origin.replace(/:8082\b/, ':8081'));
      push(origin.replace(/:8082\b/, ':8000'));
    }
    if (/:\b8080\b/.test(origin)) {
      push(origin.replace(/:8080\b/, ':8081'));
      push(origin.replace(/:8080\b/, ':8000'));
    }
    if (/^https?:\/\/(localhost|127\.0\.0\.1)(?::(?!8081)\d+)?\/?$/.test(origin)) {
      push(`${location.protocol}//${location.hostname}:8081`);
      push(`${location.protocol}//${location.hostname}:8000`);
    }
  } catch {}
  push('http://localhost:8081');
  push('http://127.0.0.1:8081');
  push('http://localhost:8000');
  push('http://127.0.0.1:8000');
  return out;
};

function parseYouTubeId(input: string): string | null {
  try {
    const url = new URL(input);
    if (url.hostname.includes('youtube.com')) {
      const v = url.searchParams.get('v');
      if (v) return v;
      const parts = url.pathname.split('/').filter(Boolean);
      const idx = parts.indexOf('embed');
      if (idx >= 0 && parts[idx + 1]) return parts[idx + 1];
    }
    if (url.hostname === 'youtu.be') {
      const id = url.pathname.replace('/', '').trim();
      return id || null;
    }
  } catch (_e) {
    if (/^[a-zA-Z0-9_-]{6,}$/.test(input)) return input;
  }
  return null;
}

export default function YoutubeLive() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState<'idle' | 'waiting' | 'captured' | 'denied' | 'cancelled' | 'noaudio' | 'stopped' | 'unsupported' | 'wrongtab' | 'stoppedByBrowser'>('idle');

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunkQueueRef = useRef<Blob[]>([]);
  const processingRef = useRef(false);
  const lastContextRef = useRef('');
  const emptyChunkCountRef = useRef(0);
  const startedSendingRef = useRef(false);
  const badChunkCountRef = useRef(0);
  const sessionIdRef = useRef('');
  const captureStartAtRef = useRef<number>(0);
  const consecutiveErrRef = useRef(0);
  const lastSegEndRef = useRef(0);
  const monitorRef = useRef<number | null>(null);
  const [captionMode, setCaptionMode] = useState<'off' | 'loading' | 'active' | 'done' | 'error'>('off');
  const captionSegmentsRef = useRef<any[]>([]);
  const captionIndexRef = useRef(0);
  const captionTimerRef = useRef<number | null>(null);
  const captionStartAtRef = useRef<number>(0);
  const ytPlayerRef = useRef<any>(null);
  const captionPollRef = useRef<number | null>(null);
  const captionKeysRef = useRef<Set<string>>(new Set());
  const captionEmittedRef = useRef<Set<string>>(new Set());

  // Gating thresholds
  const WARMUP_MS = 1500; // do not send during the first 1.5s of capture
  const MIN_CHUNK_BYTES = 16 * 1024; // drop tiny chunks

  // Merge new partial transcript text with the existing transcript while
  // removing duplicated leading content that often results from overlapping tails.
  const mergeWithDedup = (prev: string, incoming: string): string => {
    let prevText = (prev || '').trim();
    let addText = (incoming || '').trim();
    if (!prevText) return addText;
    if (!addText) return prevText;
    // Normalize whitespace for overlap check
    const prevNorm = prevText.replace(/\s+/g, ' ').trim();
    let addNorm = addText.replace(/\s+/g, ' ').trim();
    const tail = prevNorm.slice(-220); // limit comparison window
    // Char-based overlap removal
    for (let k = Math.min(tail.length, 200); k >= 30; k -= 5) {
      const suf = tail.slice(-k);
      if (addNorm.startsWith(suf)) {
        addNorm = addNorm.slice(k).trim();
        break;
      }
    }
    // Collapse extreme single-word repetitions in the addition (e.g., Savita Savita ...)
    addNorm = addNorm.replace(/\b(\w+)(?:\s+\1){2,}\b/gi, '$1');
    return addNorm ? `${prevText} ${addNorm}`.trim() : prevText;
  };

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const isTransientNetworkError = (e: any) => {
    const msg = String(e?.message || e || '').toLowerCase();
    return (
      msg.includes('failed to fetch') ||
      msg.includes('networkerror') ||
      msg.includes('load failed') ||
      msg.includes('fetch')
    );
  };

  const getVideoTimeSec = (): number => {
    try {
      const p: any = ytPlayerRef.current;
      if (p && typeof p.getCurrentTime === 'function') {
        const t = Number(p.getCurrentTime() || 0);
        if (!Number.isNaN(t) && t >= 0) return t;
      }
    } catch {}
    const base = captionStartAtRef.current || Date.now();
    return Math.max(0, (Date.now() - base) / 1000);
  };

  const getSiteSettingsUrl = () => {
    try {
      const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
      const isEdge = /Edg/i.test(ua);
      const scheme = isEdge ? 'edge' : 'chrome';
      return `${scheme}://settings/content/siteDetails?site=${encodeURIComponent(location.origin)}`;
    } catch {
      return '';
    }
  };

  const startCaptionFallback = async (id: string) => {
    try {
      if (!id) return;
      if (captionMode === 'active' || captionMode === 'loading') return;
      setCaptionMode('loading');
      // Fetch captions from backend using candidate bases
      let res: Response | null = null;
      let data: any = null;
      const bases = getApiCandidates();
      for (const base of bases) {
        try {
          const r = await fetch(`${base}/captions?video_id=${encodeURIComponent(id)}`);
          if (r.ok) { res = r; break; }
        } catch {}
      }
      if (!res) { setCaptionMode('error'); setError('This video does not provide captions.'); return; }
      try { data = await res.json(); } catch { setCaptionMode('error'); return; }
      const segs: any[] = Array.isArray(data?.segments) ? data.segments : [];
      if (!segs.length) { setCaptionMode('error'); setError('This video does not provide captions.'); return; }
      captionSegmentsRef.current = segs;
      try {
        const keys = new Set<string>();
        for (const s of segs) {
          const k = `${Math.round(((s?.start || 0) as number) * 1000)}|${String(s?.text || '').trim()}`;
          keys.add(k);
        }
        captionKeysRef.current = keys;
      } catch {}
      let nowSec = getVideoTimeSec();
      captionIndexRef.current = 0; // not used for emission anymore, kept for compatibility
      // Pre-mark any past segments as emitted so we only append from current time forward
      try {
        const emitted = new Set<string>();
        for (const s of segs) {
          const st = Number(s?.start || 0);
          const key = `${Math.round(st * 1000)}|${String(s?.text || '').trim()}`;
          if (st <= nowSec) emitted.add(key);
        }
        captionEmittedRef.current = emitted;
      } catch {}
      captionStartAtRef.current = Date.now() - nowSec * 1000;
      try { if (captionTimerRef.current) clearInterval(captionTimerRef.current); } catch {}
      captionTimerRef.current = window.setInterval(() => {
        try {
          const elapsed = getVideoTimeSec();
          const arr = captionSegmentsRef.current;
          let appended: string[] = [];
          for (let i = 0; i < arr.length; i++) {
            const s = arr[i];
            const st = Number(s?.start || 0);
            if (!(st <= elapsed)) continue;
            const key = `${Math.round(st * 1000)}|${String(s?.text || '').trim()}`;
            if (captionEmittedRef.current.has(key)) continue;
            const m = Math.floor(st / 60);
            const sec = Math.floor(st % 60);
            const ts = `${m.toString().padStart(2,'0')}:${sec.toString().padStart(2,'0')}`;
            appended.push(`[${ts}] ${String(s.text || '').trim()}`);
            captionEmittedRef.current.add(key);
          }
          if (appended.length) {
            setTranscript((prev) => prev ? `${prev}\n${appended.join('\n')}` : appended.join('\n'));
          }
          if (captionEmittedRef.current.size >= arr.length) {
            setCaptionMode('done');
            if (captionTimerRef.current) clearInterval(captionTimerRef.current);
            captionTimerRef.current = null;
          }
        } catch {}
      }, 500) as any;
      setCaptionMode('active');
    } catch {
      setCaptionMode('error');
    }
  };

  const startCaptionPolling = (id: string) => {
    try { if (captionPollRef.current) clearInterval(captionPollRef.current); } catch {}
    captionPollRef.current = window.setInterval(async () => {
      try {
        if (!id) return;
        let res: Response | null = null;
        const bases = getApiCandidates();
        for (const base of bases) {
          try {
            const r = await fetch(`${base}/captions?video_id=${encodeURIComponent(id)}`);
            if (r.ok) { res = r; break; }
          } catch {}
        }
        if (!res) return;
        const data = await res.json().catch(() => null);
        const segs: any[] = Array.isArray(data?.segments) ? data.segments : [];
        if (!segs.length) {
          setCaptionMode((m) => (m === 'off' ? 'error' : m));
          setError('This video does not provide captions.');
          return;
        }
        let added = 0;
        const keys = captionKeysRef.current;
        for (const s of segs) {
          const k = `${Math.round(((s?.start || 0) as number) * 1000)}|${String(s?.text || '').trim()}`;
          if (!keys.has(k)) {
            keys.add(k);
            captionSegmentsRef.current.push(s);
            added += 1;
          }
        }
        if (added) {
          captionSegmentsRef.current.sort((a: any, b: any) => (a?.start || 0) - (b?.start || 0));
        }
        if (captionMode !== 'active' && segs.length && !captionTimerRef.current) {
          await startCaptionFallback(id);
        }
      } catch {}
    }, 8000) as any;
  };

  useEffect(() => {
    return () => {
      if (isCapturing) {
        stopCapture();
      }
      try { if (captionTimerRef.current) clearInterval(captionTimerRef.current); } catch {}
      captionTimerRef.current = null;
      try { if (captionPollRef.current) clearInterval(captionPollRef.current); } catch {}
      captionPollRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    try {
      const w: any = window as any;
      const init = () => {
        try {
          const id = 'yt-embed-player';
          const el = document.getElementById(id);
          if (!el) return;
          if (ytPlayerRef.current) return;
          if (!w?.YT || !w.YT.Player) return;
          ytPlayerRef.current = new w.YT.Player(id, {});
        } catch {}
      };
      if ((w?.YT && w.YT.Player)) {
        init();
      } else {
        const sid = 'yt-iframe-api';
        if (!document.getElementById(sid)) {
          const tag = document.createElement('script');
          tag.src = 'https://www.youtube.com/iframe_api';
          tag.id = sid;
          document.head.appendChild(tag);
        }
        (w as any).onYouTubeIframeAPIReady = () => init();
      }
    } catch {}
  }, [videoId]);

  const loadVideo = () => {
    setError(null);
    setStatus('idle');
    const id = parseYouTubeId(youtubeUrl.trim());
    if (!id) {
      setError('Enter a valid YouTube URL or video ID');
      return;
    }
    setVideoId(id);
    void startCaptionFallback(id);
    startCaptionPolling(id);
  };

  const startCapture = async () => {
    try {
      setError(null);
      setTranscript('');
      try { chunkQueueRef.current = []; } catch {}
      lastContextRef.current = '';
      emptyChunkCountRef.current = 0;
      startedSendingRef.current = false;
      badChunkCountRef.current = 0;
      consecutiveErrRef.current = 0;
      captureStartAtRef.current = Date.now();
      lastSegEndRef.current = 0;
      try {
        const w: any = window as any;
        const sid = (w?.crypto && typeof w.crypto.randomUUID === 'function')
          ? w.crypto.randomUUID()
          : Math.random().toString(36).slice(2);
        sessionIdRef.current = sid;
      } catch { sessionIdRef.current = Math.random().toString(36).slice(2); }
      // Capability detection (prefer Chrome/Edge); don't hard-block purely on feature checks.
      const hasDisplay = !!(navigator.mediaDevices && (navigator.mediaDevices as any).getDisplayMedia);
      const hasRecorder = typeof (window as any).MediaRecorder !== 'undefined';
      // We proceed and rely on try/catch below; only error out if actual calls fail.
      if (!videoId) {
        const id = parseYouTubeId(youtubeUrl.trim());
        if (!id) {
          setError('Enter a valid YouTube URL or video ID');
          return;
        }
        setVideoId(id);
      }

      // Stop any previous capture before requesting a new one
      if (isCapturing || streamRef.current) {
        stopCapture();
      }

      setStatus('waiting');
      let displayStream: MediaStream | null = null;
      // Resolve getDisplayMedia across implementations
      const resolveGetDisplayMedia = () => {
        const md: any = (navigator as any).mediaDevices;
        if (md && typeof md.getDisplayMedia === 'function') return md.getDisplayMedia.bind(md);
        const legacy = (navigator as any).getDisplayMedia;
        if (typeof legacy === 'function') return legacy.bind(navigator);
        return null;
      };
      const gdm = resolveGetDisplayMedia();
      if (!gdm) {
        setStatus('unsupported');
        setError(null);
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
        return;
      }
      // Verify MediaRecorder and supported audio container before proceeding
      const hasMR = typeof (window as any).MediaRecorder !== 'undefined';
      if (!hasMR) {
        setStatus('unsupported');
        setError(null);
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
        return;
      }
      const canWebm = !!(window as any).MediaRecorder?.isTypeSupported?.('audio/webm;codecs=opus') ||
        !!(window as any).MediaRecorder?.isTypeSupported?.('audio/webm');
      const canOgg = !!(window as any).MediaRecorder?.isTypeSupported?.('audio/ogg;codecs=opus');
      if (!canWebm && !canOgg) {
        setStatus('unsupported');
        setError(null);
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
        return;
      }
      try {
        displayStream = await gdm({
          video: { /* @ts-ignore */ preferCurrentTab: true },
          audio: true,
        } as any);
      } catch (err: any) {
        const name = err?.name || '';
        if (name === 'OverconstrainedError' || name === 'TypeError') {
          // Retry with simpler constraints via the same resolved function
          displayStream = await gdm({ video: true, audio: true } as any);
        } else {
          throw err;
        }
      }

      const vTracks = displayStream.getVideoTracks();
      const vTrack = vTracks && vTracks[0];
      const vSettings: any = vTrack && (vTrack.getSettings ? vTrack.getSettings() : {});
      const surface = (vSettings && (vSettings as any).displaySurface) || '';
      if (surface && String(surface).toLowerCase() !== 'browser') {
        setStatus('wrongtab');
        setError('Do not select Entire Screen or Window. Please select “This Tab”.');
        displayStream.getTracks().forEach(t => t.stop());
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
        return;
      }

      const audioTracks = displayStream.getAudioTracks();
      if (!audioTracks || audioTracks.length === 0) {
        setStatus('noaudio');
        setError(null);
        displayStream.getTracks().forEach(t => t.stop());
        return;
      }

      // Ensure the audio track is live and enabled (not muted/disabled), otherwise treat as noaudio.
      const hasLiveEnabled = audioTracks.some((t: any) => {
        const live = t && t.readyState === 'live';
        const enabled = t && t.enabled !== false;
        const notMuted = t && (typeof t.muted === 'boolean' ? !t.muted : true);
        return live && enabled && notMuted;
      });
      if (!hasLiveEnabled) {
        setStatus('noaudio');
        setError(null);
        displayStream.getTracks().forEach(t => t.stop());
        return;
      }

      const audioOnly = new MediaStream();
      audioTracks.forEach(t => audioOnly.addTrack(t));

      const mimeCandidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
      ];
      const mimeType = mimeCandidates.find((m) => (window as any).MediaRecorder?.isTypeSupported?.(m));
      let mr: MediaRecorder;
      try {
        const opts = mimeType ? { mimeType } : {};
        mr = new MediaRecorder(audioOnly, opts as any);
      } catch (_e) {
        setStatus('unsupported');
        setError(null);
        displayStream.getTracks().forEach(t => t.stop());
        return;
      }
      mr.ondataavailable = (e: any) => {
        if (!e.data) return;
        const t = (e.data as any).type || '';
        if (!t || !t.startsWith('audio/')) {
          // Allow a few bad blobs before declaring noaudio
          badChunkCountRef.current += 1;
          if (badChunkCountRef.current >= 3) {
            setStatus('noaudio');
            setError(null);
            stopCapture(true);
          }
          return;
        }
        // Warmup: skip sending for the first WARMUP_MS to avoid partial container issues
        if (Date.now() - captureStartAtRef.current < WARMUP_MS) {
          return;
        }
        // Drop tiny/silent chunks without error
        if (!e.data || e.data.size < MIN_CHUNK_BYTES) {
          emptyChunkCountRef.current += 1;
          return;
        }
        const handle = async () => {
          try {
            if (!startedSendingRef.current) {
              const buf = await e.data.slice(0, 8).arrayBuffer();
              const u = new Uint8Array(buf);
              const isOgg = u.length >= 4 && u[0] === 0x4f && u[1] === 0x67 && u[2] === 0x67 && u[3] === 0x53; // OggS
              const isWebm = u.length >= 4 && u[0] === 0x1a && u[1] === 0x45 && u[2] === 0xdf && u[3] === 0xa3; // EBML
              if (!(isOgg || isWebm)) {
                badChunkCountRef.current += 1;
                if (badChunkCountRef.current >= 3) {
                  setStatus('noaudio');
                  setError(null);
                  stopCapture(true);
                }
                return;
              }
              startedSendingRef.current = true;
            }
            emptyChunkCountRef.current = 0;
            chunkQueueRef.current.push(e.data);
            void processQueue();
          } catch {
            badChunkCountRef.current += 1;
            if (badChunkCountRef.current >= 3) {
              setStatus('noaudio');
              setError(null);
              stopCapture(true);
            }
          }
        };
        void handle();
      };
      mr.onerror = () => setError('Recorder error occurred.');
      mr.start(1000);

      streamRef.current = displayStream;
      recorderRef.current = mr;
      setTranscript('');
      setIsCapturing(true);
      setStatus('captured');

      try {
        if (monitorRef.current) clearInterval(monitorRef.current);
      } catch {}
      monitorRef.current = window.setInterval(() => {
        try {
          const tracks = (streamRef.current?.getAudioTracks?.() || []) as any[];
          const ok = tracks.some((t: any) => (t?.readyState === 'live') && (t?.enabled !== false) && (typeof t?.muted === 'boolean' ? !t.muted : true));
          if (!ok) {
            stopCapture(true);
            setStatus('stoppedByBrowser');
            setError(null);
            const id = videoId || parseYouTubeId(youtubeUrl.trim());
            if (id) void startCaptionFallback(id);
          }
        } catch {}
      }, 1500) as any;

      // If user stops sharing or audio is muted/permission revoked, pause and prompt to re-enable tab audio
      try {
        const allTracks = displayStream.getTracks();
        allTracks.forEach((t) => {
          // @ts-ignore
          t.onended = () => {
            stopCapture(true);
            setStatus('stoppedByBrowser');
            setError(null);
            const id = videoId || parseYouTubeId(youtubeUrl.trim());
            if (id) void startCaptionFallback(id);
          };
        });
        audioTracks.forEach((t: any) => {
          // Some browsers fire onmute when audio capture is disabled mid-stream
          // @ts-ignore
          t.onmute = () => {
            stopCapture(true);
            setStatus('stoppedByBrowser');
            setError(null);
            const id = videoId || parseYouTubeId(youtubeUrl.trim());
            if (id) void startCaptionFallback(id);
          };
        });
      } catch {}
    } catch (e: any) {
      const name = e?.name || '';
      // Log full error for debugging different environments (Linux portals, policy blocks, etc.)
      // This helps diagnose when users report only a generic permission error.
      // eslint-disable-next-line no-console
      console.error('[YouTubeLive] getDisplayMedia failed:', { name: e?.name, message: e?.message, error: e });
      if (name === 'NotAllowedError') {
        setStatus('denied');
        setError(null);
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
      } else if (name === 'AbortError' || name === 'NotFoundError') {
        setStatus('cancelled');
        setError(null);
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
      } else {
        setStatus('stopped');
        setError(e?.message || 'Failed to start capture.');
        const id = videoId || parseYouTubeId(youtubeUrl.trim());
        if (id) void startCaptionFallback(id);
      }
    }
  };

  const stopCapture = (keepStatus?: boolean) => {
    try {
      recorderRef.current?.stop();
    } catch {}
    recorderRef.current = null;

    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}
    streamRef.current = null;

    try {
      if (monitorRef.current) {
        clearInterval(monitorRef.current);
      }
    } catch {}
    monitorRef.current = null;

    try {
      chunkQueueRef.current = [];
    } catch {}
    sessionIdRef.current = '';

    setIsCapturing(false);
    if (!keepStatus) {
      setStatus('stopped');
    }
  };

  const processQueue = async () => {
    if (processingRef.current) return;
    processingRef.current = true;
    setSending(true);
    try {
      while (isCapturing || chunkQueueRef.current.length > 0) {
        const blob = chunkQueueRef.current.shift();
        if (!blob) break;
        // Drop invalid/tiny blobs without affecting capture
        if (!(blob.type || '').startsWith('audio/')) {
          badChunkCountRef.current += 1;
          if (badChunkCountRef.current >= 3) {
            setStatus('noaudio');
            setError(null);
            stopCapture(true);
          }
          continue;
        }
        if (blob.size < MIN_CHUNK_BYTES) {
          emptyChunkCountRef.current += 1;
          continue;
        }

        const fd = new FormData();
        const ext = (blob.type || '').includes('ogg') ? 'ogg' : 'webm';
        const name = `chunk-${Date.now()}.${ext}`;
        fd.append('file', blob, name);
        fd.append('language', 'auto');
        const context = (transcript || lastContextRef.current || '').slice(-150);
        if (context) fd.append('initial_prompt', context);
        fd.append('mode', 'normal');
        if (sessionIdRef.current) fd.append('sid', sessionIdRef.current);

        const sendOnce = async () => {
          let res: Response | null = null;
          let lastErr: any = null;
          const bases = getApiCandidates();
          for (const base of bases) {
            try {
              res = await fetch(`${base}/transcribe-chunk`, { method: 'POST', body: fd });
              if (res && res.ok) {
                RESOLVED_API = base;
                break;
              }
              if (res && res.status === 400) {
                RESOLVED_API = base;
                break;
              }
              // Not ok and not 400: try next candidate
              res = null as any;
              continue;
            } catch (e: any) {
              lastErr = e;
              continue;
            }
          }
          if (!res) throw lastErr || new Error('Failed to connect to backend');
          if (res.status === 400) {
            const data = await res.json().catch(() => null);
            if (data && (data.error === 'no_audio' || /no audio/i.test(data.message || ''))) {
              setStatus('noaudio');
              setError(null);
              stopCapture(true);
              const id = videoId || parseYouTubeId(youtubeUrl.trim());
              if (id) void startCaptionFallback(id);
              return 'abort';
            }
          }
          if (res.status >= 400 && res.status < 500) {
            // Any other client-side validation/timing issue for this chunk — skip silently
            return 'skip';
          }
          if (res.status === 413 || res.status === 415 || res.status === 422 || res.status === 429) {
            // Payload/format/rate limiting issues for this chunk — skip without surfacing an error
            return 'skip';
          }
          if (res.status >= 500 || res.status === 503) {
            // Treat server-side transient/engine failures as non-fatal for this chunk
            return 'skip';
          }
          if (!res.ok) {
            const raw = await res.text().catch(() => '');
            throw new Error(raw || 'Chunk transcription failed');
          }
          const data = await res.json();
          let appendedAny = false;
          const segs: any[] = Array.isArray(data?.segments) ? data.segments : [];
          if (segs.length) {
            const fmt = (s: number) => {
              const m = Math.floor(s / 60);
              const sec = Math.floor(s % 60);
              return `${m.toString().padStart(2,'0')}:${sec.toString().padStart(2,'0')}`;
            };
            const lines: string[] = [];
            for (const s of segs) {
              const end = typeof s?.end === 'number' ? s.end : Number(s?.end) || 0;
              if (!(end > lastSegEndRef.current + 0.01)) continue;
              let marked = '';
              const words: any[] = Array.isArray(s?.words) ? s.words : [];
              if (words.length) {
                marked = words.map((w) => {
                  const token = (w?.word || '').toString();
                  return w?.low ? `[?${token}?]` : token;
                }).join('').trim();
              } else {
                marked = (s?.text || '').toString().trim();
              }
              const ts = fmt(typeof s?.start === 'number' ? s.start : Number(s?.start) || 0);
              lines.push(`[${ts}] ${marked}`);
              lastSegEndRef.current = Math.max(lastSegEndRef.current, end);
            }
            if (lines.length) {
              appendedAny = true;
              setTranscript((prev) => {
                const next = prev ? `${prev}\n${lines.join('\n')}` : lines.join('\n');
                lastContextRef.current = next.slice(-400);
                return next;
              });
            }
          }
          if (!appendedAny) {
            const newText: string = (data?.text || '').trim();
            if (newText) {
              setTranscript((prev) => {
                const appended = mergeWithDedup(prev, newText);
                lastContextRef.current = appended.slice(-400);
                return appended;
              });
            }
          }
          // Clear any previous transient error on successful response
          setError(null);
          return 'ok';
        };

        let sent = false;
        for (let attempt = 0; attempt < 3 && !sent; attempt++) {
          try {
            const r = await sendOnce();
            if (r === 'abort') { sent = true; break; }
            if (r === 'skip') { sent = true; consecutiveErrRef.current = 0; setError(null); break; }
            sent = true;
            consecutiveErrRef.current = 0;
          } catch (e: any) {
            if (isTransientNetworkError(e) && attempt < 2) {
              // Backend could be reloading; wait and retry without surfacing error
              await sleep(600 * (attempt + 1));
              continue;
            }
            // Only surface error after multiple consecutive failures; otherwise continue
            consecutiveErrRef.current += 1;
            if (consecutiveErrRef.current >= 3) {
              setError(e?.message || 'Chunk transcription failed');
            }
            // Give backend a moment before proceeding to next chunk
            await sleep(300);
            break;
          }
        }
      }
    } finally {
      processingRef.current = false;
      setSending(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl gradient-primary" />
            <span className="font-display font-semibold text-foreground">YouTube Live Transcription</span>
          </div>
        </div>
      </header>

      <main className="container max-w-5xl mx-auto px-4 py-8 space-y-8">
        <section className="space-y-4">
          <div className="flex flex-col md:flex-row gap-3">
            <div className="flex-1 flex items-center gap-2">
              <LinkIcon className="w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Paste YouTube URL or ID"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                disabled={isCapturing}
              />
              <Button variant="secondary" onClick={loadVideo} disabled={!youtubeUrl || isCapturing}>
                Load
              </Button>
            </div>
            
            <Button variant="outline" disabled={!videoId} asChild>
              <a
                href={videoId ? `https://www.youtube.com/watch?v=${videoId}` : '#'}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open on YouTube
              </a>
            </Button>
          </div>
          <div className="text-xs text-muted-foreground -mt-1">
            <ul className="list-disc pl-5 space-y-1">
              <li>Load a YouTube video to transcribe from its available captions.</li>
              <li>No microphone, tab audio, or screen permissions are required.</li>
            </ul>
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <p className="text-xs text-muted-foreground">Using YouTube captions only. No permissions required.</p>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="aspect-video bg-black/5 border border-border rounded-xl overflow-hidden">
            {videoId ? (
              <iframe
                id="yt-embed-player"
                className="w-full h-full"
                src={`https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1&enablejsapi=1&origin=${location.origin}`}
                title="YouTube video player"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">
                Paste a YouTube URL and click Load
              </div>
            )}
          </div>

          <div className="border border-border rounded-xl p-4 flex flex-col min-h-[280px]">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-display font-semibold">Live transcript</h3>
              {captionMode !== 'off' && (
                <span className="text-xs text-muted-foreground">YouTube captions</span>
              )}
              {sending && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Processing…
                </div>
              )}
            </div>
            <div className="flex-1 overflow-auto whitespace-pre-wrap text-sm text-foreground/90">
              {transcript || (
                <span className="text-muted-foreground">Chunks will appear here as you play the video…</span>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
