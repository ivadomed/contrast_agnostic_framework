import React from 'react';
import {
  AbsoluteFill, Img, Sequence, Audio, interpolate, spring, useCurrentFrame, useVideoConfig,
  staticFile, Easing, delayRender, continueRender,
} from 'remotion';
import manifest from '../public/assets/manifest.json';
import TL from './timeline.json';

/* ------------------------------------------------------------------ look */
const C = {
  bg: '#06080b', panel: '#0d1117', ink: '#eef2f6', muted: '#8a96a6', line: '#1e2733',
  amber: '#ffb547', cyan: '#5fd3e8', red: '#ff6b5b',
};
const DISPLAY = '"Bricolage", "Helvetica Neue", Arial, sans-serif';
const MONO = '"PlexMono", Menlo, Consolas, monospace';
const SANS = '"PlexSans", "Helvetica Neue", Arial, sans-serif';

const fontHandle = delayRender('fonts');
Promise.all([
  new FontFace('Bricolage', `url(${staticFile('fonts/bricolage.woff2')})`, {weight: '200 800'}).load(),
  new FontFace('PlexMono', `url(${staticFile('fonts/plexmono.woff2')})`, {weight: '400'}).load(),
  new FontFace('PlexSans', `url(${staticFile('fonts/plexsans.woff2')})`, {weight: '100 700'}).load(),
]).then((fs) => { fs.forEach((f) => (document.fonts as any).add(f)); continueRender(fontHandle); })
  .catch(() => continueRender(fontHandle));

const A = (p: string) => staticFile('assets/' + p);
const SOURCE: Record<string, string> = {
  brain_healthy: 'ON-Harmony', brain_tumor: 'BraTS 2024', brain_ms: 'Open-MS', abdomen: 'CHAOS',
  breast: 'I-SPY2', mandible: 'ToothFairy2', pelvis: 'TotalSegmentator',
};
type Task = {key: string; name: string; contrasts: {name: string; file: string}[]; draws: number};
const TASKS = (manifest as any).tasks as Task[];
const HERO = (manifest as any).hero as {key: string; k: number; regions: number; probe: {x: number; y: number; mu: number; alpha: number; mean: number}};
const TEX = (manifest as any).texture as {key: string; name: string; zoom: {cx: number; cy: number; half: number}}[];
const task = (k: string) => TASKS.find((t) => t.key === k)!;

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const ease = Easing.bezier(0.22, 1, 0.36, 1);
const fmt = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);

/* ------------------------------------------------------------------ timeline (shared with the soundtrack) */
type SceneKey = 'mosaic' | 'problem' | 'transform' | 'gallery' | 'draws' | 'bench' | 'texture' | 'ablation' | 'end';
const S = {} as Record<SceneKey, [number, number]>;
let acc = 0;
for (const [k, d] of TL.scenes as [SceneKey, number][]) { S[k] = [acc, d]; acc += d; }
export const TOTAL = acc;
const BEAT = (TL.fps * 60) / TL.bpm;                       // 15 frames
// 1 on every kick, decaying over the beat (scenes start on beats, so local frames keep the phase)
const beatPulse = (f: number) => Math.exp(-((((f % BEAT) + BEAT) % BEAT) / 3.5));

/* enter with a quick zoom + blur, leave with a short scale-down fade */
const SceneTransition: React.FC<{dur: number; children: React.ReactNode}> = ({dur, children}) => {
  const f = useCurrentFrame();
  const inP = interpolate(f, [0, 10], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const outP = interpolate(f, [dur - 8, dur], [0, 1], {...clamp, easing: Easing.in(Easing.cubic)});
  const scale = (1.07 - 0.07 * inP) * (1 - 0.03 * outP);
  const blur = 10 * (1 - inP) + 6 * outP;
  return (
    <AbsoluteFill style={{opacity: inP * (1 - outP), transform: `scale(${scale})`, filter: blur > 0.05 ? `blur(${blur}px)` : undefined}}>
      {children}
    </AbsoluteFill>
  );
};

const Kicker: React.FC<{children: React.ReactNode; color?: string}> = ({children, color = C.amber}) => (
  <div style={{fontFamily: MONO, fontSize: 22, letterSpacing: '0.14em', textTransform: 'uppercase', color}}>{children}</div>
);

const Frame: React.FC<{src: string; size: number; style?: React.CSSProperties; children?: React.ReactNode}> = ({src, size, style, children}) => (
  <div style={{position: 'relative', width: size, height: size, background: '#000', borderRadius: 10, overflow: 'hidden', ...style}}>
    <Img src={src} style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}} />
    {children}
  </div>
);

/* ------------------------------------------------------------------ 1. mosaic */
const Mosaic: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const tiles: {src: string; label: string}[] = [];
  TASKS.forEach((t) => t.contrasts.forEach((c) => tiles.push({src: A(c.file), label: c.name})));
  const grid = Array.from({length: 24}, (_, i) => tiles[(i * 7) % tiles.length]);
  const order = grid.map((_, i) => (i * 11) % 24);
  const zoom = interpolate(f, [0, 150], [1.0, 1.12]);
  const titleIn = spring({frame: f - TL.mosaicTitle, fps, config: {damping: 200}});
  return (
    <AbsoluteFill style={{background: C.bg}}>
      <AbsoluteFill style={{transform: `scale(${zoom})`, display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 10, padding: 10}}>
        {grid.map((t, i) => {
          const s = spring({frame: f - order[i] * 2, fps, config: {damping: 14, stiffness: 120}});
          return (
            <div key={i} style={{position: 'relative', aspectRatio: '1', opacity: s, transform: `scale(${0.55 + 0.45 * s})`, borderRadius: 8, overflow: 'hidden', background: '#000'}}>
              <Img src={t.src} style={{width: '100%', height: '100%'}} />
              <div style={{position: 'absolute', left: 10, bottom: 8, fontFamily: MONO, fontSize: 14, color: '#c9d3df', textShadow: '0 1px 3px #000'}}>{t.label}</div>
            </div>
          );
        })}
      </AbsoluteFill>
      <AbsoluteFill style={{background: `rgba(6,8,11,${0.72 * titleIn})`}} />
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: titleIn, transform: `scale(${0.92 + 0.08 * titleIn})`}}>
        <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 190, color: C.ink, letterSpacing: '-0.03em'}}>PALETTE-Aug</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ network diagram */
// Classic fully connected sketch: columns of nodes, every node linked to the next column.
// `p` in [0,1] is the position of the forward pass (null = idle); one layer is orange at a time
// and the links light up while the signal travels between two layers.
const LAYERS = [4, 6, 7, 6, 4];
const NeuralNet: React.FC<{w: number; h: number; p: number | null}> = ({w, h, p}) => {
  const L = LAYERS.length;
  const r = Math.min(h / 18, 16);
  const pos = p === null ? -10 : p * (L - 1);
  const nodes = LAYERS.map((m, l) => Array.from({length: m}, (_, k) => ({
    x: r + ((w - 2 * r) * l) / (L - 1), y: h / 2 + (k - (m - 1) / 2) * Math.min((h - 2 * r) / (Math.max(...LAYERS) - 1), 3.2 * r),
  })));
  const idle = '#2c3746';
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{overflow: 'visible'}}>
      {nodes.slice(0, -1).map((col, l) => {
        const on = pos > l && pos < l + 1 ? 1 - Math.abs(pos - l - 0.5) * 2 : 0;
        return col.map((a, i) => nodes[l + 1].map((b, j) => (
          <line key={`${l}-${i}-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={on > 0 ? `rgba(255,181,71,${0.25 + 0.55 * on})` : idle} strokeWidth={on > 0 ? 2.2 : 1.4} />
        )));
      })}
      {nodes.map((col, l) => {
        const on = Math.abs(pos - l) < 0.5;
        return col.map((a, i) => (
          <circle key={`n${l}-${i}`} cx={a.x} cy={a.y} r={r} fill={on ? C.amber : '#141b24'} stroke={on ? C.amber : '#4a5768'} strokeWidth={2.5} />
        ));
      })}
    </svg>
  );
};

const Arrow: React.FC<{w: number; p: number | null}> = ({w, p}) => (
  <svg width={w} height={40} viewBox={`0 0 ${w} 40`}>
    <line x1={6} y1={20} x2={w - 18} y2={20} stroke="#3a4656" strokeWidth={3} />
    <path d={`M ${w - 22} 9 L ${w - 4} 20 L ${w - 22} 31 Z`} fill="#3a4656" />
    {p !== null && p > 0 && p < 1 && <circle cx={6 + (w - 24) * p} cy={20} r={7} fill={C.amber} />}
  </svg>
);

/* ------------------------------------------------------------------ 2. problem: segment the training contrast, "?" on the unseen ones */
const Mask: React.FC<{src: string; p: number}> = ({src, p}) => (
  <div style={{position: 'absolute', inset: 0, background: C.amber, opacity: 0.72, clipPath: `circle(${p * 75}% at 50% 50%)`,
    WebkitMaskImage: `url(${src})`, WebkitMaskSize: '100% 100%', maskImage: `url(${src})`, maskSize: '100% 100%'} as React.CSSProperties} />
);

const Problem: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = task('brain_tumor');
  const PR = TL.problem;
  const IMG = 330, UW = 420, UH = 300, AW = 110, CAP = 50, GAP = 60;
  const ROW = IMG + CAP;                                  // the network travels exactly one row down
  const ramp = (a: number, b: number) => interpolate(f, [a, b], [0, 1], clamp);
  const label = (txt: string, col: string) => (
    <div style={{width: 190, fontFamily: MONO, fontSize: 24, letterSpacing: '0.14em', color: col}}>{txt}</div>
  );
  // 1) train on T1n: forward pass, prediction (ground-truth mask)
  const in1 = spring({frame: f, fps, config: {damping: 200}});
  const pass1 = f >= PR.pass1[0] && f <= PR.pass1[1] ? (f - PR.pass1[0]) / (PR.pass1[1] - PR.pass1[0]) : null;
  const pred = ramp(PR.pred, PR.pred + 22);
  const badge = spring({frame: f - PR.badge, fps, config: {damping: 200}});
  // 2) the SAME trained network moves down to the unseen contrasts
  const in2 = spring({frame: f - PR.row2, fps, config: {damping: 200}});
  const move = interpolate(f, PR.move, [0, 1], {...clamp, easing: Easing.inOut(Easing.cubic)});
  // 3) unseen contrasts go through it, one after another: no segmentation
  const SEG = PR.seg, start2 = PR.start2;
  const seg = Math.max(0, Math.min(2, Math.floor((f - start2) / SEG)));
  const u = f - start2 - seg * SEG;
  const unseen = t.contrasts.slice(1);
  const pass2 = f >= start2 && u >= PR.pass2[0] && u <= PR.pass2[1] ? (u - PR.pass2[0]) / (PR.pass2[1] - PR.pass2[0]) : null;
  const q = f >= start2 ? spring({frame: u - PR.pass2[1], fps, config: {damping: 11, stiffness: 150}}) : 0;
  const fadeIn = f >= start2 ? ramp(start2 + seg * SEG, start2 + seg * SEG + 6) : 0;
  const cur = f >= start2 ? unseen[seg] : unseen[0], prev = seg > 0 && f >= start2 ? unseen[seg - 1] : null;
  const row = (children: React.ReactNode, o: number) => (
    <div style={{display: 'flex', alignItems: 'flex-start', gap: 22, height: ROW, opacity: o, transform: `translateY(${(1 - o) * 50}px)`}}>{children}</div>
  );
  const centred = (h: number, node: React.ReactNode) => <div style={{height: IMG, display: 'flex', alignItems: 'center'}}>{node}</div>;
  const caption = (txt: string, col = C.ink) => <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 34, color: col, marginTop: 10, textAlign: 'center'}}>{txt}</div>;
  const netP = move < 1 ? (pass1 !== null ? pass1 : null) : pass2;
  return (
    <AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center', gap: GAP}}>
      {row(<>
        {centred(IMG, label('TRAINING', C.amber))}
        <div><Frame src={A(`${t.key}/input.png`)} size={IMG} style={{outline: `4px solid ${C.amber}`, outlineOffset: 5}} />{caption(t.contrasts[0].name, C.amber)}</div>
        {centred(IMG, <Arrow w={AW} p={pass1 === null ? null : Math.min(1, pass1 * 3)} />)}
        <div style={{position: 'relative', width: UW, height: IMG, zIndex: 2}}>
          {/* one network: trained above, then carried down to the unseen row */}
          <div style={{position: 'absolute', left: 0, top: (IMG - UH) / 2, transform: `translateY(${move * (ROW + GAP)}px)`}}>
            <NeuralNet w={UW} h={UH} p={netP} />
            <div style={{position: 'absolute', left: 0, right: 0, top: UH + 14, textAlign: 'center', fontFamily: MONO, fontSize: 22,
              letterSpacing: '0.1em', color: C.amber, opacity: badge}}>trained on T1n</div>
          </div>
          {/* faint outline where the network was */}
          <div style={{position: 'absolute', inset: 0, border: `2px dashed ${C.line}`, borderRadius: 12, opacity: move * 0.8}} />
        </div>
        {centred(IMG, <Arrow w={AW} p={pass1 === null ? null : Math.max(0, pass1 * 3 - 2)} />)}
        <div><Frame src={A(`${t.key}/input.png`)} size={IMG} style={{opacity: pred}}><Mask src={A(`${t.key}/labels.png`)} p={pred} /></Frame>{caption('prediction', `rgba(238,242,246,${pred})`)}</div>
      </>, in1)}
      {row(<>
        {centred(IMG, label('UNSEEN', C.muted))}
        <div>
          <div style={{position: 'relative', width: IMG, height: IMG}}>
            {prev && <Frame src={A(prev.file)} size={IMG} style={{position: 'absolute'}} />}
            {cur && <Frame src={A(cur.file)} size={IMG} style={{position: 'absolute', opacity: f >= start2 ? fadeIn : 1}} />}
          </div>
          {caption(cur ? cur.name : '')}
        </div>
        {centred(IMG, <Arrow w={AW} p={pass2 === null ? null : Math.min(1, pass2 * 3)} />)}
        <div style={{width: UW, height: IMG}} />
        {centred(IMG, <Arrow w={AW} p={pass2 === null ? null : Math.max(0, pass2 * 3 - 2)} />)}
        <div>
          <Frame src={cur ? A(cur.file) : A(`${t.key}/input.png`)} size={IMG}>
            <div style={{position: 'absolute', inset: 0, background: 'rgba(6,8,11,0.72)'}} />
            <div style={{position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', fontFamily: DISPLAY, fontWeight: 700, fontSize: 190,
              color: C.ink, opacity: q, transform: `scale(${0.5 + 0.5 * q})`}}>?</div>
          </Frame>
          {caption("can't segment", `rgba(201,102,91,${q})`)}
        </div>
      </>, in2)}
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ 3. transform */
const STEPS = [{t: 'Input'}, {t: 'k-means'}, {t: 'Voronoi'}, {t: 'Label remap'}, {t: 'Affine remap'}];
const STEP_AT = TL.transformSteps;   // in choreography frames; played TL.transformSpeed x faster

const Reveal: React.FC<{src: string; p: number; mode: 'circle' | 'wipe'; cx?: number; cy?: number; opacity?: number}> = ({src, p, mode, cx = 50, cy = 50, opacity = 1}) => {
  const clip = mode === 'circle' ? `circle(${p * 75}% at ${cx}% ${cy}%)` : `inset(0 0 ${(1 - p) * 100}% 0)`;
  return <Img src={src} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', clipPath: clip, opacity}} />;
};

const Transform: React.FC = () => {
  const f = useCurrentFrame() * TL.transformSpeed;   // warped time: the whole choreography plays faster
  const {fps} = useVideoConfig();
  const k = HERO.key;
  const step = STEP_AT.filter((s) => f >= s).length - 1;
  const ramp = (a: number, b: number) => interpolate(f, [a, b], [0, 1], {...clamp, easing: ease});
  // overlays: show partition map, then cross to the grayscale flat remap
  const clsIn = ramp(98, 128), clsOut = ramp(160, 185);
  const ridIn = ramp(208, 240), ridOut = ramp(270, 295);
  const labIn = ramp(318, 350), labOut = ramp(380, 405);
  const sweep = ramp(TL.transformSweep[0], TL.transformSweep[1]);
  const probeIn = spring({frame: f - 505, fps, config: {damping: 200}});
  const size = 860;
  const P = HERO.probe;
  const eqLines = [
    `y = μ + α (x − x̄)`,
    `μ = ${fmt(P.mu)}   α = ${fmt(P.alpha)}   x̄ = ${fmt(P.mean)}`,
  ];
  const typed = Math.floor(interpolate(f, [520, 580], [0, eqLines[1].length], clamp));
  return (
    <AbsoluteFill style={{background: C.bg, flexDirection: 'row', alignItems: 'center', padding: '0 110px', gap: 80}}>
      <div style={{position: 'relative', width: size, height: size, background: '#000', borderRadius: 12, overflow: 'hidden', flexShrink: 0}}>
        <Img src={A(`${k}/hero_a.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}} />
        <Img src={A(`${k}/hero_b.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: clsOut}} />
        <Reveal src={A(`${k}/hero_cls.png`)} p={clsIn} mode="circle" opacity={1 - clsOut} />
        <Img src={A(`${k}/hero_c.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: ridOut}} />
        <Reveal src={A(`${k}/hero_rid.png`)} p={ridIn} mode="circle" cx={P.x * 100} cy={P.y * 100} opacity={1 - ridOut} />
        <Img src={A(`${k}/hero_d.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: labOut}} />
        <Reveal src={A(`${k}/labels.png`)} p={labIn} mode="wipe" opacity={(1 - labOut) * 0.95} />
        <Reveal src={A(`${k}/hero_e.png`)} p={sweep} mode="wipe" />
        {sweep > 0 && sweep < 1 && (
          <div style={{position: 'absolute', left: 0, right: 0, top: `${sweep * 100}%`, height: 3, background: C.amber, boxShadow: `0 0 24px 6px rgba(255,181,71,0.55)`}} />
        )}
        <div style={{position: 'absolute', left: `${P.x * 100}%`, top: `${P.y * 100}%`, width: 46, height: 46, margin: '-23px 0 0 -23px', borderRadius: '50%', border: `3px solid ${C.amber}`, opacity: probeIn, transform: `scale(${2 - probeIn})`}} />
        <div style={{position: 'absolute', left: 18, top: 14, fontFamily: MONO, fontSize: 17, color: '#8fa3bb', lineHeight: 1.4}}>
          {SOURCE[k]} · T1w<br />k = {HERO.k} · {HERO.regions} regions
        </div>
        <div style={{position: 'absolute', right: 18, bottom: 14, fontFamily: MONO, fontSize: 17, color: '#8fa3bb'}}>({'abcde'[step]})</div>
      </div>

      <div style={{display: 'grid', gap: 34, flex: 1}}>
        <div>
          <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 72, color: C.amber, lineHeight: 1}}>PALETTE-Aug</div>
          <div style={{fontFamily: MONO, fontSize: 24, color: C.muted, marginTop: 12}}>what it does to one training image</div>
        </div>
        <div style={{display: 'grid', gap: 26}}>
          {STEPS.map((s, i) => {
            const on = i === step, done = i < step;
            return (
              <div key={i} style={{display: 'grid', gridTemplateColumns: '56px 1fr', alignItems: 'start', opacity: on ? 1 : done ? 0.55 : 0.28}}>
                <div style={{width: 40, height: 40, borderRadius: 20, display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 20,
                  background: on ? C.amber : 'transparent', color: on ? C.bg : C.muted, border: `2px solid ${on ? C.amber : C.line}`}}>{'abcde'[i]}</div>
                <div style={{fontFamily: DISPLAY, fontSize: on ? 56 : 44, fontWeight: 700, color: C.ink, lineHeight: 1.1}}>{s.t}</div>
              </div>
            );
          })}
        </div>
        <div style={{fontFamily: MONO, fontSize: 30, color: C.ink, padding: '22px 26px', border: `1px solid ${C.line}`, borderRadius: 10, opacity: interpolate(f, [440, 470], [0, 1], clamp), lineHeight: 1.6}}>
          <div>{eqLines[0]}</div>
          <div style={{color: C.amber}}>{eqLines[1].slice(0, typed)}<span style={{opacity: f % 30 < 15 ? 1 : 0}}>▍</span></div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ 4. training: one task, one new draw per iteration */
// One model, one task (the BraTS T1n case of the previous scene). Each iteration: a new augmented
// T1n, a forward pass, a prediction; the counter steps by one exactly when the image changes.
// Period eases from P0 to P1 frames (P1 = 10 -> at most 3 image changes per second, crossfaded).
const {P0, P1, RAMP, xf: XF} = TL.draws;
const iterCount = (g: number) => {  // integral of 1/period
  const gg = Math.min(g, RAMP);
  const n = -(RAMP / (P0 - P1)) * Math.log((P0 - ((P0 - P1) * gg) / RAMP) / P0);
  return n + Math.max(0, g - RAMP) / P1;
};
const periodAt = (g: number) => P0 - ((P0 - P1) * Math.min(g, RAMP)) / RAMP;
const drawSrc = (t: Task, idx: number) =>
  idx < 0 ? A(`${t.key}/input.png`) : A(`${t.key}/draw_${String(idx % t.draws).padStart(2, '0')}.png`);

const Crossfade: React.FC<{a: string; b: string; mix: number; fit?: 'contain' | 'fill'}> = ({a, b, mix, fit = 'contain'}) => (
  <>
    <Img src={a} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: fit}} />
    {mix > 0 && <Img src={b} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: fit, opacity: mix}} />}
  </>
);

// illustrative, deterministic, decaying training loss (not from a real log)
const lossAt = (k: number) => 0.18 + 0.72 * Math.exp(-k / 5) + 0.05 * Math.exp(-k / 20) * Math.sin(k * 2.3) * Math.cos(k * 0.7);

const LossCurve: React.FC<{k: number; kMax: number; w: number; h: number}> = ({k, kMax, w, h}) => {
  const pts: string[] = [];
  for (let j = 0; j <= k; j++) pts.push(`${(j / kMax) * w},${h - lossAt(j) * h}`);
  const last = pts[pts.length - 1].split(',').map(Number);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{overflow: 'visible'}}>
      <line x1={0} y1={h} x2={w} y2={h} stroke={C.line} strokeWidth={2} />
      <line x1={0} y1={0} x2={0} y2={h} stroke={C.line} strokeWidth={2} />
      <polyline points={pts.join(' ')} fill="none" stroke={C.amber} strokeWidth={3} strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r={6} fill={C.amber} />
      <text x={8} y={-12} fill={C.muted} fontFamily={MONO} fontSize={20}>loss</text>
    </svg>
  );
};

const Draws: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = task('brain_tumor');
  const g = Math.max(0, f - TL.draws.start);
  const started = g > 0;
  const n = started ? iterCount(g) : 0;
  const K = Math.floor(n), frac = n - K, period = periodAt(g);
  const el = frac * period;                          // frames into the current iteration
  const kMax = Math.ceil(iterCount(S.draws[1] - TL.draws.start));
  const mix = started ? Math.min(1, el / XF) : 1;
  const pass = started ? Math.min(1, el / (TL.draws.pass * period)) : null;
  const predOn = started ? interpolate(el, [TL.draws.pred[0] * period, TL.draws.pred[1] * period], [0, 1], clamp) : 0;
  const IMG = 470, NW = 520, NH = 380, AW = 105;
  const appear = spring({frame: f, fps, config: {damping: 200}});
  // filmstrip of finished iterations, newest on the left, sliding right as each one finishes
  const TH = 128, TG = 14, slide = started ? interpolate(el, [0, 6], [0, 1], {...clamp, easing: ease}) : 1;
  const past = Array.from({length: 8}, (_, j) => K - 1 - j).filter((k) => k >= 0);
  return (
    <AbsoluteFill style={{background: C.bg, padding: '60px 70px', gap: 56, justifyContent: 'center', opacity: appear}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
        <div>
          <div style={{position: 'relative', width: IMG, height: IMG, borderRadius: 10, overflow: 'hidden', background: '#000', outline: `4px solid ${C.amber}`, outlineOffset: 5}}>
            <Crossfade a={drawSrc(t, K - 1)} b={drawSrc(t, started ? K : -1)} mix={mix} />
          </div>
          <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 32, color: C.amber, marginTop: 14, textAlign: 'center'}}>augmented T1n</div>
        </div>
        <Arrow w={AW} p={pass === null ? null : Math.min(1, pass * 4)} />
        <NeuralNet w={NW} h={NH} p={pass !== null && pass < 1 ? pass : null} />
        <Arrow w={AW} p={pass === null ? null : Math.max(0, pass * 4 - 3)} />
        <div>
          <Frame src={drawSrc(t, started ? K : -1)} size={IMG} style={{opacity: started ? 1 : 0.3}}>
            <Mask src={A(`${t.key}/labels.png`)} p={predOn} />
          </Frame>
          <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 32, color: C.ink, marginTop: 14, textAlign: 'center'}}>prediction</div>
        </div>
      </div>
      <div style={{display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between'}}>
        <div style={{position: 'relative', width: 8 * (TH + TG), height: TH, overflow: 'hidden'}}>
          {past.map((k, j) => (
            <div key={k} style={{position: 'absolute', left: (j - 1 + slide) * (TH + TG), top: 0, width: TH, height: TH, borderRadius: 6, overflow: 'hidden',
              background: '#000', opacity: 1 - j * 0.08}}>
              <Img src={drawSrc(t, k)} style={{width: '100%', height: '100%'}} />
            </div>
          ))}
        </div>
        <div style={{display: 'flex', alignItems: 'flex-end', gap: 50}}>
          <div style={{textAlign: 'right'}}>
            <div style={{fontFamily: MONO, fontSize: 20, letterSpacing: '0.14em', color: C.muted}}>ITERATION</div>
            <div style={{fontFamily: MONO, fontSize: 72, lineHeight: 1, color: C.amber, fontVariantNumeric: 'tabular-nums'}}>{started ? K + 1 : 0}</div>
          </div>
          <LossCurve k={started ? K : 0} kMax={kMax} w={300} h={120} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ 3b. gallery: every task, a new draw over and over */
// Not training: just PALETTE-Aug applied repeatedly to one real slice per task. Each tile has its
// own phase (staggered start), first change is an amber scan, then a crossfade every `period` frames.
const Gallery: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const G = TL.gallery;
  const push = interpolate(f, [0, S.gallery[1]], [1, 1.05]);
  const kick = beatPulse(f);
  const SZ = 400;
  return (
    <AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center'}}>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 24, width: 4 * SZ + 3 * 24, transform: `scale(${push})`}}>
        {TASKS.map((t, i) => {
          const s = spring({frame: f - i * 3, fps, config: {damping: 14, stiffness: 120}});
          const t0 = G.t0 + i * G.stagger, e = f - t0 - G.scan;
          const scan = interpolate(f, [t0, t0 + G.scan], [0, 1], {...clamp, easing: ease});
          const k = e >= 0 ? Math.floor(e / G.period) : -1, el = e - k * G.period;
          const mix = e >= 0 ? interpolate(el, [G.period - G.xf, G.period], [0, 1], clamp) : 0;
          return (
            <div key={t.key} style={{position: 'relative', width: SZ, height: SZ, borderRadius: 10, overflow: 'hidden', background: '#000',
              opacity: s, transform: `scale(${(0.7 + 0.3 * s) * (1 + 0.015 * kick)})`}}>
              {e < 0 ? (
                <>
                  <Img src={A(`${t.key}/input.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}} />
                  <Img src={drawSrc(t, i * 5)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', clipPath: `inset(0 0 ${(1 - scan) * 100}% 0)`}} />
                  {scan > 0 && scan < 1 && <div style={{position: 'absolute', left: 0, right: 0, top: `${scan * 100}%`, height: 3, background: C.amber, boxShadow: '0 0 18px 4px rgba(255,181,71,.55)'}} />}
                </>
              ) : (
                <Crossfade a={drawSrc(t, i * 5 + k)} b={drawSrc(t, i * 5 + k + 1)} mix={mix} fit="fill" />
              )}
              <div style={{position: 'absolute', left: 14, bottom: 10, fontFamily: SANS, fontWeight: 600, fontSize: 22, color: C.ink, textShadow: '0 1px 4px #000'}}>{t.name}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ 5. texture */
const TexturePanel: React.FC<{t: (typeof TEX)[number]; delay: number}> = ({t, delay}) => {
  const f = useCurrentFrame() - delay;
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 200}});
  const z = interpolate(f, [20, 90], [0, 1], {...clamp, easing: ease});
  const scale = 1 + z * (0.5 / t.zoom.half - 1);
  const tx = (0.5 - t.zoom.cx) * 100 * z, ty = (0.5 - t.zoom.cy) * 100 * z;
  const split = 50 + 34 * Math.sin(Math.max(0, f - 95) / 34) * interpolate(f, [95, 120], [0, 1], clamp);
  const img = (file: string) => (
    <Img src={A(`${t.key}/${file}`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', transformOrigin: '50% 50%', transform: `scale(${scale}) translate(${tx}%, ${ty}%)`}} />
  );
  return (
    <div style={{display: 'grid', gap: 18, opacity: s, transform: `translateY(${(1 - s) * 60}px)`}}>
      <div style={{position: 'relative', width: 720, height: 720, borderRadius: 12, overflow: 'hidden', background: '#000'}}>
        {img('tex_noise.png')}
        <div style={{position: 'absolute', inset: 0, clipPath: `inset(0 0 0 ${split}%)`}}>{img('tex_real.png')}</div>
        <div style={{position: 'absolute', top: 0, bottom: 0, left: `${split}%`, width: 3, background: C.amber, boxShadow: '0 0 18px rgba(255,181,71,.6)'}} />
        <div style={{position: 'absolute', left: 18, top: 14, fontFamily: MONO, fontSize: 19, color: C.muted, background: 'rgba(0,0,0,.55)', padding: '4px 10px', borderRadius: 4}}>noise</div>
        <div style={{position: 'absolute', right: 18, top: 14, fontFamily: MONO, fontSize: 19, color: C.amber, background: 'rgba(0,0,0,.55)', padding: '4px 10px', borderRadius: 4}}>real texture</div>
      </div>
      <div style={{fontFamily: DISPLAY, fontSize: 38, fontWeight: 700, color: C.ink}}>{t.name}<span style={{fontFamily: MONO, fontSize: 18, color: C.muted, fontWeight: 400, marginLeft: 16}}>{SOURCE[t.key]}</span></div>
    </div>
  );
};
const Texture: React.FC = () => (
  <AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center', gap: 34}}>
    <div style={{display: 'flex', alignItems: 'baseline', gap: 24}}>
      <span style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 64, color: C.ink}}>Ablation</span>
      <span style={{fontFamily: MONO, fontSize: 22, color: C.muted}}>same regions · noise vs. real texture</span>
    </div>
    <div style={{display: 'flex', gap: 80}}>
      {TEX.map((t, i) => <TexturePanel key={t.key} t={t} delay={i * 10} />)}
    </div>
  </AbsoluteFill>
);

/* ------------------------------------------------------------------ 6. benchmark + ablation bars */
// Numbers from the paper (do not edit by hand without re-checking the sources):
//  BENCH: paper tab:meta (= paper/generated_results/meta_task_heatmap_paper_summary.md, 2026-10-07, SRCSM with its
//         test-time source matching): Dice, PALETTE-Aug val000 minus the best OTHER method per task (PALETTE alone
//         excluded); mark = that best-other method's own cell (★ Ours significantly better, ▼ significantly worse).
//  ABL:   paper/scripts/compute_dissociation_pvalues.py task rows (2026-10-09): relative Δ Dice at the fill swap
//         (rung 4 noise fill -> rung 5 PALETTE alone, val000), ★/▼ = Holm p < 0.05 over the 7 tasks. Rung 4 = the
//         label_voronoi noise-fill retrain everywhere except Brain, whose retrain had not landed (previous rung 4 kept).
type Bar = {key: string | null; label: string; v: number; mark: '★' | '▼' | ''};
const BENCH: Bar[] = [
  {key: 'brain_tumor', label: 'Glioma', v: 1.1, mark: '★'},
  {key: 'brain_ms', label: 'MS', v: 4.2, mark: '★'},
  {key: 'breast', label: 'Breast', v: 1.1, mark: '★'},
  {key: null, label: 'Spine', v: 1.4, mark: '★'},
  {key: 'pelvis', label: 'Pelvis', v: 1.3, mark: '★'},
  {key: 'abdomen', label: 'Abdomen', v: 0.3, mark: ''},
  {key: 'brain_healthy', label: 'Brain', v: 0.2, mark: ''},
  {key: 'mandible', label: 'Mandible', v: -2.1, mark: ''},
];
const ABL: Bar[] = [
  {key: 'brain_ms', label: 'MS', v: 37.8, mark: '★'},
  {key: 'brain_tumor', label: 'Glioma', v: 13.0, mark: '★'},
  {key: 'breast', label: 'Breast', v: 4.5, mark: '★'},
  {key: 'pelvis', label: 'Pelvis', v: 6.3, mark: '★'},
  {key: 'abdomen', label: 'Abdomen', v: 4.2, mark: '★'},
  {key: 'mandible', label: 'Mandible', v: -2.2, mark: ''},
  {key: 'brain_healthy', label: 'Brain', v: -1.1, mark: '▼'},
];

const BarChart: React.FC<{bars: Bar[]; unit: string; vmax: number; vmin: number; groups?: [string, number][]}> = ({bars, unit, vmax, vmin, groups}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const H = 520, zero = (H * vmax) / (vmax - vmin), px = H / (vmax - vmin);
  const colW = 170, gap = 34;
  return (
    <div style={{display: 'flex', gap, alignItems: 'flex-end'}}>
      {bars.map((b, i) => {
        const s = spring({frame: f - TL.bars.delay - i * TL.bars.step, fps, config: {damping: 16, stiffness: 90}});
        const h = Math.abs(b.v) * px * s, pos = b.v >= 0;
        const win = b.v > 0;
        const col = b.mark === '▼' ? '#c9665b' : win ? C.amber : '#4a5563';
        const star = spring({frame: f - TL.bars.star - i * TL.bars.step, fps, config: {damping: 10, stiffness: 160}});
        const extraGap = groups && i === groups[0][1] ? 70 : 0;
        return (
          <div key={i} style={{width: colW, marginLeft: extraGap, display: 'grid', justifyItems: 'center', gap: 16}}>
            <div style={{position: 'relative', width: colW, height: H}}>
              <div style={{position: 'absolute', left: -gap / 2, right: -gap / 2, top: zero, height: 2, background: C.line}} />
              <div style={{position: 'absolute', left: 30, right: 30, top: pos ? zero - h : zero + 2, height: h, background: col, borderRadius: 4}} />
              <div style={{position: 'absolute', left: 0, right: 0, textAlign: 'center', top: pos ? zero - h - 92 : zero + h + 14, opacity: s,
                fontFamily: MONO, fontSize: 32, color: col === '#4a5563' ? C.muted : col, fontVariantNumeric: 'tabular-nums'}}>
                <div style={{fontSize: 44, height: 46, transform: `scale(${star})`, color: b.mark === '▼' ? '#c9665b' : C.amber}}>{b.mark}</div>
                {(b.v > 0 ? '+' : b.v < 0 ? '−' : '') + Math.abs(b.v).toFixed(1) + unit}
              </div>
            </div>
            <div style={{width: 128, height: 128, borderRadius: 64, overflow: 'hidden', background: '#000', border: `3px solid ${win ? C.amber : C.line}`, display: 'grid', placeItems: 'center'}}>
              {b.key ? <Img src={A(`${b.key}/input.png`)} style={{width: '100%', height: '100%'}} /> :
                <span style={{fontFamily: MONO, fontSize: 18, color: C.muted}}>no image</span>}
            </div>
            <div style={{fontFamily: SANS, fontWeight: 600, fontSize: 26, color: C.ink}}>{b.label}</div>
          </div>
        );
      })}
    </div>
  );
};

const Bench: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  // the counter ticks as each winning bar lands
  const n = BENCH.filter((b, i) => b.v > 0 && f > TL.bars.delay + i * TL.bars.step + 18).length;  // tasks where Ours has the best Dice
  return (
    <AbsoluteFill style={{background: C.bg, padding: '70px 90px', flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between'}}>
      <BarChart bars={BENCH} unit="" vmax={5} vmin={-4.4} />
      <div style={{position: 'absolute', left: 90, top: 60}}>
        <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 84, color: C.ink, lineHeight: 1}}>Results</div>
        <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, marginTop: 12}}>Δ Dice vs. best other method · ★ significant</div>
      </div>
      <div style={{position: 'absolute', right: 90, top: 60, textAlign: 'right'}}>
        <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 170, color: C.amber, lineHeight: 0.9, fontVariantNumeric: 'tabular-nums'}}>{n}<span style={{color: C.muted}}>/8</span></div>
        <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, marginTop: 12}}>best Dice</div>
      </div>
    </AbsoluteFill>
  );
};

const Ablation: React.FC = () => (
  <AbsoluteFill style={{background: C.bg, padding: '70px 90px', justifyContent: 'flex-end'}}>
    <div style={{position: 'absolute', left: 90, top: 70, fontFamily: MONO, fontSize: 24, letterSpacing: '0.12em', color: C.amber}}>APPEARANCE</div>
    <div style={{position: 'absolute', left: 90 + 3 * 170 + 3 * 34 + 70, top: 70, fontFamily: MONO, fontSize: 24, letterSpacing: '0.12em', color: C.muted}}>INTERFACE</div>
    <div style={{position: 'absolute', right: 90, top: 56, textAlign: 'right'}}>
      <div style={{fontFamily: DISPLAY, fontWeight: 700, fontSize: 84, color: C.ink, lineHeight: 1}}>Ablation results</div>
      <div style={{fontFamily: MONO, fontSize: 22, color: C.muted, marginTop: 12}}>noise → real texture · relative Δ Dice · ★ significant</div>
    </div>
    <BarChart bars={ABL} unit="%" vmax={40} vmin={-14} groups={[['interface', 3]]} />
  </AbsoluteFill>
);

/* ------------------------------------------------------------------ 7. end card */
const End: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 200}});
  const strip = TASKS.map((t, i) => {
    const n = f / 18 + i / TASKS.length, k = Math.floor(n);
    return {a: drawSrc(t, k + i * 3), b: drawSrc(t, k + 1 + i * 3), mix: interpolate(n - k, [0.8, 1], [0, 1], clamp)};
  });
  return (
    <AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center', gap: 50}}>
      <div style={{textAlign: 'center', opacity: s, transform: `scale(${0.94 + 0.06 * s})`}}>
        <div style={{fontFamily: DISPLAY, fontSize: 168, fontWeight: 700, color: C.ink, letterSpacing: '-0.03em', lineHeight: 1}}>PALETTE-Aug</div>
      </div>
      <div style={{display: 'flex', gap: 14, opacity: interpolate(f, [15, 40], [0, 0.85], clamp)}}>
        {strip.map((x, i) => <div key={i} style={{position: 'relative', width: 150, height: 150, borderRadius: 8, overflow: 'hidden', background: '#000'}}><Crossfade a={x.a} b={x.b} mix={x.mix} fit="fill" /></div>)}
      </div>
      <div style={{fontFamily: MONO, fontSize: 17, color: C.muted, opacity: interpolate(f, [25, 45], [0, 1], clamp)}}>
        Data: ON-Harmony · BraTS 2024 · Open-MS · CHAOS · I-SPY2 · ToothFairy2 · TotalSegmentator
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ main */
export const Main: React.FC = () => (
  <AbsoluteFill style={{background: C.bg}}>
    <Audio src={staticFile('audio/soundtrack.wav')} />
    {([
      ['mosaic', Mosaic], ['problem', Problem], ['transform', Transform], ['gallery', Gallery], ['draws', Draws],
      ['bench', Bench], ['texture', Texture], ['ablation', Ablation], ['end', End],
    ] as const).map(([k, Comp]) => {
      const [from, dur] = S[k];
      return (
        <Sequence key={k} from={from} durationInFrames={dur}>
          <SceneTransition dur={dur}><Comp /></SceneTransition>
        </Sequence>
      );
    })}
  </AbsoluteFill>
);
