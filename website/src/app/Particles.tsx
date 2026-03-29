"use client";
import { useEffect, useRef } from "react";

const N = 450;

interface Dot {
  x: number; y: number;
  tx: number; ty: number;
  size: number; tsize: number;
  r: number; g: number; b: number;
  tr: number; tg: number; tb: number;
  alpha: number; talpha: number;
}

function sphere(cx: number, cy: number, r: number, n: number): [number, number][] {
  const pts: [number, number][] = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const inc = Math.acos(1 - 2 * t);
    const az = ga * i;
    pts.push([cx + r * Math.sin(inc) * Math.cos(az), cy + r * Math.sin(inc) * Math.sin(az)]);
  }
  while (pts.length < N) pts.push([cx, cy]);
  return pts;
}

export default function Particles() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const sublabelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0, h = 0, cx = 0, cy = 0;
    const dots: Dot[] = [];
    const seed = Array.from({ length: N * 4 }, () => Math.random());
    const startTime = performance.now();

    function resize() {
      if (!canvas) return;
      w = window.innerWidth; h = window.innerHeight;
      cx = w / 2; cy = h / 2;
      canvas.width = w * 2; canvas.height = h * 2;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      ctx!.setTransform(2, 0, 0, 2, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < N; i++) {
      dots.push({
        x: w/2, y: h/2, tx: w/2, ty: h/2,
        size: 0, tsize: 0,
        r: 26, g: 26, b: 26, tr: 26, tg: 26, tb: 26,
        alpha: 0, talpha: 0,
      });
    }

    function setLabel(main: string, sub: string) {
      if (labelRef.current) { labelRef.current.textContent = main; labelRef.current.style.opacity = main ? "1" : "0"; }
      if (sublabelRef.current) { sublabelRef.current.textContent = sub; sublabelRef.current.style.opacity = sub ? "1" : "0"; }
    }

    function set(i: number, x: number, y: number, s: number, a: number, r: number, g: number, b: number) {
      if (i >= N) return;
      dots[i].tx = x; dots[i].ty = y;
      dots[i].tsize = s; dots[i].talpha = a;
      dots[i].tr = r; dots[i].tg = g; dots[i].tb = b;
    }

    function hideAll() {
      for (let i = 0; i < N; i++) { dots[i].tsize = 0; dots[i].talpha = 0; }
    }

    const P: [number,number,number] = [139,92,246];
    const G: [number,number,number] = [34,197,94];
    const A: [number,number,number] = [245,158,11];
    const K: [number,number,number] = [236,72,153];
    const B: [number,number,number] = [40,40,50];

    const scenes: { dur: number; main: string; sub: string; drawConn: boolean; setup: () => void }[] = [

      // 0: Single dot
      { dur: 2100, main: "", sub: "", drawConn: false, setup: () => {
        hideAll();
        set(0, cx, cy, 6, 1, ...P);
      }},

      // 1: Install
      { dur: 2100, main: "You install Samhati on your laptop.", sub: "One command. Any device with a GPU — even a MacBook Air.", drawConn: false,  setup: () => {
        hideAll();
        for (let i = 0; i < 40; i++) {
          const a = (i/40) * Math.PI * 2;
          const r = 15 + seed[i] * 25;
          set(i, cx + Math.cos(a)*r, cy + Math.sin(a)*r, 1.5 + seed[i]*2, 0.8, ...B);
        }
        set(0, cx, cy, 7, 1, ...P);
      }},

      // 2: Model selected
      { dur: 2100, main: "Your node picks a model — Qwen 3B fits in 4GB RAM.", sub: "llama.cpp starts. Proof of Inference enabled. You're live.", drawConn: false,  setup: () => {
        hideAll();
        for (let i = 0; i < 50; i++) {
          const a = (i/50) * Math.PI * 2;
          const r = 30 + seed[i] * 20;
          set(i, cx + Math.cos(a)*r, cy + Math.sin(a)*r, 2 + seed[i]*1.5, 0.8, ...(i < 15 ? G : B));
        }
        set(0, cx, cy, 7, 1, ...G);
      }},

      // 3: Global nodes
      { dur: 2400, main: "Laptops, desktops, servers — joining from everywhere.", sub: "Tokyo runs Qwen-14B. São Paulo has DeepSeek-Coder. Berlin serves Math.", drawConn: false,  setup: () => {
        hideAll();
        const cities: [number,number][] = [
          [cx-320, cy-140], [cx+300, cy-110], [cx-220, cy+160],
          [cx+340, cy+130], [cx-50, cy-210], [cx-380, cy+40],
          [cx+120, cy+210], [cx+400, cy-30], [cx-150, cy-50], [cx, cy],
        ];
        for (let i = 0; i < 180; i++) {
          const c = cities[i % cities.length];
          const colors = [G, A, K, B, G, B, A, K, G, B] as const;
          const col = colors[i % cities.length];
          set(i, c[0]+(seed[i*2]-0.5)*60, c[1]+(seed[i*2+1]-0.5)*60, 1.5+seed[i]*2, 0.75, ...col);
        }
      }},

      // 4: Mesh connects
      { dur: 2400, main: "Solana registry + iroh gossip. They find each other.", sub: "QUIC connections. NAT traversal. No manual IP entry needed.", drawConn: true,  setup: () => {
        hideAll();
        const colors = [G, A, K, B] as const;
        const pts = sphere(cx, cy, Math.min(w,h)*0.3, 280);
        for (let i = 0; i < 280; i++) {
          set(i, pts[i][0], pts[i][1], 2+seed[i]*1.5, 0.8, ...colors[i%4]);
        }
      }},

      // 5: Dashboard demand
      { dur: 2400, main: "The Dashboard shows what the network needs.", sub: "Code 42% · Math 16% · Reasoning 11% · General 31%", drawConn: false,  setup: () => {
        hideAll();
        const domains = [
          { pct: 0.42, c: G, y: cy - 90 },
          { pct: 0.16, c: A, y: cy - 30 },
          { pct: 0.11, c: K, y: cy + 30 },
          { pct: 0.31, c: B, y: cy + 90 },
        ];
        let idx = 0;
        for (const d of domains) {
          const count = Math.floor(d.pct * 120);
          const barW = d.pct * w * 0.5;
          for (let j = 0; j < count && idx < N; j++, idx++) {
            set(idx, cx - w*0.25 + (j/count)*barW, d.y + (seed[idx]-0.5)*8, 2.5, 0.85, ...d.c);
          }
        }
      }},

      // 6: Specialist bonus
      { dur: 2600, main: "Specialists earn 1.5× SMTI on matched queries.", sub: "Run what the network needs → earn more tokens.", drawConn: false,  setup: () => {
        hideAll();
        const domains = [
          { pct: 0.42, c: G, y: cy - 90 },
          { pct: 0.16, c: A, y: cy - 30 },
          { pct: 0.11, c: K, y: cy + 30 },
          { pct: 0.31, c: B, y: cy + 90 },
        ];
        let idx = 0;
        for (const d of domains) {
          const count = Math.floor(d.pct * 120);
          const barW = d.pct * w * 0.5;
          const isSpec = d.c !== B;
          for (let j = 0; j < count && idx < N; j++, idx++) {
            set(idx, cx - w*0.25 + (j/count)*barW, d.y, isSpec ? 3.5 : 1.5, isSpec ? 1 : 0.3, ...d.c);
          }
        }
      }},

      // 7: Query arrives
      { dur: 2400, main: "Someone asks: \"Write a binary search in Rust.\"", sub: "Complexity: Medium · Domain: Code", drawConn: false,  setup: () => {
        hideAll();
        const pts = sphere(cx, cy, Math.min(w,h)*0.32, 250);
        for (let i = 5; i < 250; i++) set(i, pts[i][0], pts[i][1], 1, 0.1, ...B);
        set(0, cx, cy, 10, 1, ...P);
      }},

      // 8: Fan-out
      { dur: 2600, main: "Fan-out → 3 Code specialist nodes.", sub: "Each runs llama.cpp + generates PoI proof.", drawConn: false,  setup: () => {
        hideAll();
        const pts = sphere(cx, cy, Math.min(w,h)*0.32, 250);
        for (let i = 5; i < 250; i++) set(i, pts[i][0], pts[i][1], 1, 0.08, ...B);
        const targets: [number,number][] = [[cx-200, cy-130],[cx+180, cy-90],[cx-80, cy+160]];
        for (let i = 0; i < 3; i++) set(i, targets[i][0], targets[i][1], 7, 1, ...G);
        set(3, cx, cy, 4, 0.5, ...P);
      }},

      // 9: Peer ranking
      { dur: 2600, main: "Nodes peer-rank each other's answers.", sub: "Pairwise comparisons. 50-100 token reasoning chains.", drawConn: false,  setup: () => {
        hideAll();
        for (let i = 0; i < 3; i++) {
          const a = (i/3) * Math.PI * 2;
          set(i, cx + Math.cos(a)*70, cy + Math.sin(a)*70, 6, 1, ...G);
        }
        for (let i = 3; i < 60; i++) {
          const a = (i/60) * Math.PI * 6;
          const r = 40 + (i/60) * 80;
          set(i, cx + Math.cos(a)*r, cy + Math.sin(a)*r, 1, 0.3, ...P);
        }
      }},

      // 10: Winner
      { dur: 2100, main: "BradleyTerry picks the winner. +16 ELO.", sub: "Best answer shown to user. [Code | 3 nodes | 94% confidence | 1.8s]", drawConn: false,  setup: () => {
        hideAll();
        set(0, cx, cy, 14, 1, ...P);
        set(1, cx-150, cy-80, 3, 0.25, ...B);
        set(2, cx+140, cy+90, 3, 0.25, ...B);
        for (let i = 3; i < 120; i++) {
          const a = seed[i*2] * Math.PI * 2;
          const progress = seed[i*2+1];
          const r = 250 * (1 - progress * progress);
          set(i, cx + Math.cos(a)*r, cy + Math.sin(a)*r, 1+progress*3, 0.3+progress*0.7, ...A);
        }
      }},

      // 11: Settlement
      { dur: 2100, main: "Round settles on Solana. Proof hashes go on-chain.", sub: "NodeAccount updated. RoundAccount created. SMTI distributed.", drawConn: false,  setup: () => {
        hideAll();
        const cols = 15, rows = 10;
        const gw = w * 0.5, gh = h * 0.5;
        let idx = 0;
        for (let r = 0; r < rows && idx < N; r++) {
          for (let c = 0; c < cols && idx < N; c++, idx++) {
            const x = cx - gw/2 + (c/(cols-1)) * gw;
            const y = cy - gh/2 + (r/(rows-1)) * gh;
            const hl = (r === 4 || r === 5) && c > 5 && c < 12;
            set(idx, x, y, hl ? 3.5 : 2, hl ? 1 : 0.3, ...(hl ? P : B));
          }
        }
      }},

      // 12: Network grows
      { dur: 2600, main: "The network grows. More nodes. More intelligence. No gatekeepers.", sub: "Open intelligence for everyone, forever.", drawConn: true,  setup: () => {
        hideAll();
        const colors = [G, A, K, B, P] as const;
        const pts = sphere(cx, cy, Math.min(w,h)*0.4, N);
        for (let i = 0; i < N; i++) {
          set(i, pts[i][0], pts[i][1], 2+seed[i]*2, 0.85, ...colors[i%5]);
        }
      }},

      // 13: Collapse → loop
      { dur: 1000, main: "", sub: "", drawConn: false,  setup: () => {
        hideAll();
        set(0, cx, cy, 6, 1, ...P);
      }},
    ];

    let totalDur = 0;
    const tl = scenes.map(s => {
      const entry = { start: totalDur, end: totalDur + s.dur, ...s };
      totalDur += s.dur;
      return entry;
    });

    let lastScene = -1;
    let raf: number;

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);

      const elapsed = (performance.now() - startTime) % totalDur;
      let si = 0;
      for (let i = 0; i < tl.length; i++) {
        if (elapsed >= tl[i].start && elapsed < tl[i].end) { si = i; break; }
      }

      if (si !== lastScene) {
        lastScene = si;
        cx = w/2; cy = h/2;
        tl[si].setup();
        setLabel(tl[si].main, tl[si].sub);
      }

      // Animate
      for (const d of dots) {
        d.x += (d.tx - d.x) * 0.14;
        d.y += (d.ty - d.y) * 0.14;
        d.size += (d.tsize - d.size) * 0.18;
        d.alpha += (d.talpha - d.alpha) * 0.15;
        d.r += (d.tr - d.r) * 0.15;
        d.g += (d.tg - d.g) * 0.15;
        d.b += (d.tb - d.b) * 0.15;
      }

      // Connections
      if (tl[si].drawConn) {
        ctx.lineWidth = 0.3;
        for (let i = 0; i < N; i++) {
          if (dots[i].alpha < 0.3) continue;
          for (let j = i+1; j < Math.min(i+8, N); j++) {
            if (dots[j].alpha < 0.3) continue;
            const dx = dots[i].x - dots[j].x;
            const dy = dots[i].y - dots[j].y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 50) {
              ctx.beginPath();
              ctx.moveTo(dots[i].x, dots[i].y);
              ctx.lineTo(dots[j].x, dots[j].y);
              ctx.strokeStyle = `rgba(80,80,100,${(1-dist/50)*0.15})`;
              ctx.stroke();
            }
          }
        }
      }

      // Fan-out lines (scene 8)
      if (si === 8) {
        ctx.lineWidth = 1;
        for (let i = 0; i < 3; i++) {
          ctx.beginPath();
          ctx.moveTo(dots[3].x, dots[3].y);
          ctx.lineTo(dots[i].x, dots[i].y);
          ctx.strokeStyle = `rgba(139,92,246,0.2)`;
          ctx.stroke();
        }
      }

      // Reward streams (scene 10)
      if (si === 10) {
        ctx.lineWidth = 0.4;
        for (let i = 3; i < 120; i++) {
          if (dots[i].alpha < 0.1) continue;
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y);
          ctx.lineTo(dots[0].x, dots[0].y);
          ctx.strokeStyle = `rgba(245,158,11,${dots[i].alpha * 0.06})`;
          ctx.stroke();
        }
      }

      // Ranking connections (scene 9)
      if (si === 9) {
        ctx.lineWidth = 0.6;
        for (let a = 0; a < 3; a++) {
          for (let b = a+1; b < 3; b++) {
            ctx.beginPath();
            ctx.moveTo(dots[a].x, dots[a].y);
            ctx.lineTo(dots[b].x, dots[b].y);
            ctx.strokeStyle = `rgba(34,197,94,0.25)`;
            ctx.stroke();
          }
        }
      }

      // Grid lines (scene 11)
      if (si === 11) {
        const cols = 15;
        ctx.lineWidth = 0.2;
        for (let i = 0; i < 150; i++) {
          if (dots[i].alpha < 0.1) continue;
          if ((i+1) % cols !== 0 && i+1 < 150) {
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[i+1].x, dots[i+1].y);
            ctx.strokeStyle = `rgba(80,80,100,${dots[i].alpha * 0.12})`;
            ctx.stroke();
          }
          if (i + cols < 150) {
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[i+cols].x, dots[i+cols].y);
            ctx.strokeStyle = `rgba(80,80,100,${dots[i].alpha * 0.12})`;
            ctx.stroke();
          }
        }
      }

      // Draw dots
      for (const d of dots) {
        if (d.alpha < 0.01 || d.size < 0.1) continue;
        // Glow for big dots
        if (d.size > 5) {
          ctx.beginPath();
          ctx.arc(d.x, d.y, d.size * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${Math.round(d.r)},${Math.round(d.g)},${Math.round(d.b)},${d.alpha * 0.06})`;
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${Math.round(d.r)},${Math.round(d.g)},${Math.round(d.b)},${d.alpha})`;
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    }

    draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
  }, []);

  return (
    <>
      <canvas ref={canvasRef} className="fixed inset-0" style={{ zIndex: 0 }} />
      <div ref={labelRef} className="fixed bottom-24 left-0 right-0 text-center z-10 pointer-events-none px-8 transition-opacity duration-500" style={{ fontSize: "17px", color: "#1A1A1A", fontWeight: 500 }} />
      <div ref={sublabelRef} className="fixed bottom-12 left-0 right-0 text-center z-10 pointer-events-none px-8 transition-opacity duration-500" style={{ fontSize: "13px", color: "#999", fontWeight: 300 }} />
    </>
  );
}
