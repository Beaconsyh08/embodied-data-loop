"use strict";
(() => {
  const manifest = window.UMI_MANIFEST;
  const $ = id => document.getElementById(id);
  const names = {left_arm:"左手",right_arm:"右手",head:"头部"};
  const video = $("video"), canvas = $("chart");
  let current = null, selected = 0, loadId = 0, activeScript = null;
  let frame = 0, range = [0,1], errors = [];
  function getPart(part) { const base=current.parts[part]; return $("method").value==="cosmos" ? {...base,...(base.candidates?.cosmos||{})} : base; }
  window.UMI_REVIEW_STATE = {ready:false, errors, frame:0, episode:null};
  if (!manifest) {$("status").textContent="结果索引尚未准备好，请稍后刷新。";return;}
  $("batch-summary").textContent=`${manifest.episodes.length} / ${manifest.total_episodes} 条已就绪 · ${manifest.total_frames.toLocaleString()} 帧 · ${manifest.fps} FPS`;
  if (manifest.reports?.length) {
    $("evidence-links").hidden=false;
    for (const report of manifest.reports) {const a=document.createElement("a");a.href=report.path;a.textContent=report.title;$("evidence-links").append(a);}
  }
  function renderStatus() {
    const row=manifest.episodes.find(x=>x.episode===selected), cosmos=$("method").value==="cosmos";
    $("status").className="notice"+(!cosmos&&!row.residual&&!row.regression?" good":"");
    $("status").textContent=cosmos?"Cosmos ID 试点候选：仅替换已标记异常区间；仍需逐段复核。正常观测遮蔽对照未写入此候选。":row.regression?"本条有部分数值指标恶化，请重点检查。下方保留原始和修复后实际数值。":row.residual?"修复后仍有严重数值跳变，需要继续复核。":"未检出超过本次阈值的严重残余跳变；尚未完成物理正确性与人工验收。";
    const replay=row.replay, statusName={accepted:"通过",rejected:"跟随误差超限",error:"未完成回放"};
    $("replay-result").hidden=!replay;
    if(replay)$("replay-result").textContent=`机器人回放（固定基准映射）：原始 ${statusName[replay.raw.status]} → 已有试修复 ${statusName[replay.repaired.status]}。${replay.repaired.error||replay.repaired.reasons?.join("；")||"head 单独报告，结果不等于人工验收。"}`;
    const count=Object.keys(current.parts).reduce((s,p)=>s+getPart(p).changed.reduce((n,v)=>n+v,0),0);
    $("episode-info").textContent=`${current.frames} 帧 · ${(current.frames/current.fps).toFixed(2)} 秒 · ${count} 个“帧×部位”被修改`;
  }
  function renderList() {
    $("episodes").replaceChildren();
    const filter=$("filter").value;
    for (const row of manifest.episodes) {
      if (filter==="regression"&&!row.regression || filter==="residual"&&!row.residual || filter==="clean"&&row.residual) continue;
      const button=document.createElement("button");button.className="episode-button"+(row.episode===selected?" active":"");button.dataset.episode=row.episode;
      const title=document.createElement("span");title.textContent=`Episode ${String(row.episode).padStart(2,"0")}`;
      const badge=document.createElement("span");badge.className=row.regression?"worse":row.residual?"bad":"";badge.textContent=row.regression?"部分恶化":row.residual?"残余跳变":"待复核";
      const small=document.createElement("small");small.textContent=`${row.frames} 帧 · ${row.duration.toFixed(2)} s`;
      button.append(title,badge,small);button.onclick=()=>load(row.episode);$("episodes").append(button);
    }
  }
  function load(episode) {
    video.pause();const row=manifest.episodes.find(x=>x.episode===episode);if(!row)return;
    selected=episode;const id=++loadId;current=null;window.UMI_REVIEW_STATE.ready=false;
    $("status").textContent="正在加载完整 episode…";$("episode-title").textContent=`Episode ${String(episode).padStart(2,"0")}`;
    renderList();if(activeScript)activeScript.remove();
    activeScript=document.createElement("script");activeScript.src=row.data;
    activeScript.onload=()=>{
      if(id!==loadId || window.UMI_EPISODE.episode!==episode)return;
      current=window.UMI_EPISODE;frame=0;range=[0,current.frames-1];
      $("range-start").value=0;$("range-end").value=range[1];$("range-end").max=range[1];$("range-start").max=range[1];
      $("frame-number").max=range[1];$("scrub").max=range[1];
      $("task").textContent=Array.isArray(current.tasks)?current.tasks.join(" / "):String(current.tasks||"");
      $("method").value="openpi"; $("method").querySelector("option[value=cosmos]").disabled=!row.candidate_videos?.cosmos; video.src=row.video;$("download").href=row.video;
      history.replaceState(null,"",`#episode=${episode}`);
      renderStatus();renderMetrics();renderSegments();setFrame(0,false);
      Object.assign(window.UMI_REVIEW_STATE,{ready:true,episode,frames:current.frames,method:"openpi"});
    };
    activeScript.onerror=()=>{errors.push(`episode ${episode} data unavailable`);$("status").textContent="本条数据加载失败，请确认结果目录完整。";};
    document.body.append(activeScript);
  }
  function fmt(n,d=2) {return n==null?"—":Number(n).toFixed(d);}
  function distance(a,b){return Math.hypot(...a.slice(0,3).map((v,i)=>v-b[i]))*1000;}
  function setFrame(value,seek=true) {
    if(!current)return;frame=Math.max(0,Math.min(current.frames-1,Math.round(Number(value)||0)));
    $("scrub").value=frame;$("frame-number").value=frame;$("time-label").textContent=`${frame} / ${current.frames-1} 帧 · ${fmt(current.times[frame],3)} s`;
    if(seek)video.currentTime=(frame+.1)/current.fps;
    window.UMI_REVIEW_STATE.frame=frame;drawChart();renderFrame();
  }
  function renderFrame() {
    $("frame-table").replaceChildren();
    for(const part of Object.keys(current.parts)){ const p=getPart(part);
      const tr=document.createElement("tr");
      const values=[names[part],p.valid[frame]==null?"unknown":p.valid[frame]?"true":"false",p.changed[frame]?"是":"否",`${fmt(distance(p.raw[frame],p.repaired[frame]))} mm`,frame?`${fmt(distance(p.raw[frame],p.raw[frame-1]))} mm`:"—",frame?`${fmt(distance(p.repaired[frame],p.repaired[frame-1]))} mm`:"—"];
      values.forEach((x,i)=>{const td=document.createElement("td");td.textContent=x;if(i===1)td.className=p.valid[frame]?"":"regression";tr.append(td);});$("frame-table").append(tr);
    }
  }
  function renderMetrics(){
    $("metrics-table").replaceChildren();
    for(const part of Object.keys(current.parts)){ const p=getPart(part);
      const tr=document.createElement("tr");const b=p.before,a=p.after;
      [names[part],`${b.steps_gt_100mm} → ${a.steps_gt_100mm}`,`${b.steps_gt_10deg} → ${a.steps_gt_10deg}`,`${fmt(b.max_step_mm)} → ${fmt(a.max_step_mm)} mm`,`${p.changed.reduce((s,v)=>s+v,0)} / ${current.frames}`].forEach((x,i)=>{const td=document.createElement("td");td.textContent=x;if((i===1&&a.steps_gt_100mm>b.steps_gt_100mm)||(i===2&&a.steps_gt_10deg>b.steps_gt_10deg))td.className="regression";tr.append(td);});$("metrics-table").append(tr);
    }
  }
  function renderSegments(){
    $("segments").replaceChildren();const p=getPart($("part").value);
    for(const s of p.segments){
      const b=document.createElement("button");b.className="segment";b.textContent=`${s.component} · ${s.start}–${s.end_exclusive-1} · ${s.method}`;
      b.onclick=()=>{range=[Math.max(0,s.start-15),Math.min(current.frames-1,s.end_exclusive+14)];$("range-start").value=range[0];$("range-end").value=range[1];setFrame(s.start);};$("segments").append(b);
    }
    if(!p.segments.length)$("segments").textContent="本 Tracker 没有报告修复区间。";
  }
  function getSeries(){
    const p=getPart($("part").value),m=$("metric").value;
    if(m==="position")return [p.raw.map(x=>x.slice(0,3)),p.repaired.map(x=>x.slice(0,3)),["X","Y","Z"],"m"];
    if(m==="rotation")return [p.raw_rotation,p.repaired_rotation,["Rx","Ry","Rz"],"°"];
    const key=m==="speed"?"speed":"angular_speed";
    return [p[`raw_${key}`].map(x=>[x]),p[`repaired_${key}`].map(x=>[x]),[m==="speed"?"线速度":"角速度"],m==="speed"?"m/s":"°/s"];
  }
  function drawChart(){
    if(!current)return;const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,w=Math.max(rect.width,300),h=rect.height;
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
    const [raw,fixed,labels,unit]=getSeries();const p=getPart($("part").value),start=range[0],end=range[1];
    let lo=Infinity,hi=-Infinity;for(const arr of [raw,fixed])for(let i=start;i<=end;i++)for(const v of arr[i])if(v!=null&&Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
    if(!Number.isFinite(lo)){lo=0;hi=1;}let pad=Math.max((hi-lo)*.08,1e-4);lo-=pad;hi+=pad;
    const box={left:72,right:w-20,top:20,bottom:h-49};
    const x=i=>box.left+(i-start)/Math.max(end-start,1)*(box.right-box.left),y=v=>box.bottom-(v-lo)/(hi-lo)*(box.bottom-box.top);
    ctx.font="11px system-ui";ctx.textBaseline="middle";
    for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,py=y(v);ctx.strokeStyle="#30405a";ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(box.left,py);ctx.lineTo(box.right,py);ctx.stroke();ctx.fillStyle="#a7b7d0";ctx.textAlign="right";ctx.fillText(Math.abs(v)>1000?v.toExponential(1):v.toFixed(Math.abs(hi-lo)<.1?3:2),box.left-8,py);}
    ctx.save();ctx.beginPath();ctx.rect(box.left,box.top,box.right-box.left,box.bottom-box.top);ctx.clip();
    const colors=["#69a8ff","#ffb25a","#c395ff"];
    for(let j=0;j<2;j++)for(let d=0;d<labels.length;d++){ctx.strokeStyle=colors[d];ctx.lineWidth=j?2:1.4;ctx.globalAlpha=j?1:.55;ctx.setLineDash(j?[]:[5,4]);ctx.beginPath();let begun=false;for(let i=start;i<=end;i++){const v=(j?fixed:raw)[i][d];if(v==null||!Number.isFinite(v)){begun=false;continue;}if(begun)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));begun=true;}ctx.stroke();}
    ctx.restore();ctx.globalAlpha=1;ctx.setLineDash([]);
    for(let i=start;i<=end;i++){const dx=Math.max(1,(box.right-box.left)/Math.max(end-start,1));if(p.valid[i]===0){ctx.fillStyle="#d6a34c";ctx.fillRect(x(i),box.bottom+8,dx,4);}if(p.changed[i]){ctx.fillStyle="#ae83dd";ctx.fillRect(x(i),box.bottom+15,dx,4);}}
    for(let i=0;i<=4;i++){const f=Math.round(start+(end-start)*i/4);ctx.fillStyle="#a7b7d0";ctx.textAlign="center";ctx.fillText(String(f),x(f),h-13);}
    if(frame>=start&&frame<=end){ctx.strokeStyle="#ffe19a";ctx.beginPath();ctx.moveTo(x(frame),box.top);ctx.lineTo(x(frame),box.bottom+20);ctx.stroke();}
    ctx.fillStyle="#a7b7d0";ctx.textAlign="left";ctx.fillText(unit,7,10);
    let lines=[];for(let d=0;d<labels.length;d++)lines.push(`${labels[d]} 原始 ${fmt(raw[frame][d],4)} → 修复 ${fmt(fixed[frame][d],4)} ${unit}`);$("cursor-values").textContent=lines.join("　 |　 ");
    window.UMI_REVIEW_STATE.chartRange=[...range];
  }
  function updateRange(){if(!current)return;let a=Math.max(0,Math.min(current.frames-2,Number($("range-start").value)||0)),b=Math.max(a+1,Math.min(current.frames-1,Number($("range-end").value)||current.frames-1));range=[a,b];$("range-start").value=a;$("range-end").value=b;drawChart();}
  $("filter").onchange=renderList;$("scrub").oninput=e=>{video.pause();setFrame(e.target.value);};$("frame-number").onchange=e=>{video.pause();setFrame(e.target.value);};
  $("previous").onclick=()=>{video.pause();setFrame(frame-1);};$("next").onclick=()=>{video.pause();setFrame(frame+1);};
  $("play").onclick=()=>{if(video.paused)video.play().catch(e=>{errors.push(String(e));});else video.pause();};
  video.onplay=()=>$("play").textContent="暂停";video.onpause=()=>$("play").textContent="播放";
  function sync(){if(current&&!video.seeking)setFrame(Math.floor(video.currentTime*current.fps+1e-3),false);}
  video.ontimeupdate=sync;video.onseeked=sync;video.onerror=()=>{errors.push(`video error episode ${selected}`);};
  if(video.requestVideoFrameCallback){const tick=(_now,meta)=>{if(current)setFrame(Math.floor(meta.mediaTime*current.fps+1e-3),false);video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick);}
  $("method").onchange=()=>{ if(!current)return;video.pause();const row=manifest.episodes.find(x=>x.episode===selected),method=$("method").value;video.src=method==="cosmos"?row.candidate_videos.cosmos:row.video;$("download").href=video.src;renderMetrics();renderSegments();setFrame(0,false);window.UMI_REVIEW_STATE.method=method;renderStatus();};
  $("part").onchange=()=>{renderSegments();drawChart();};$("metric").onchange=drawChart;
  $("range-start").onchange=updateRange;$("range-end").onchange=updateRange;$("reset-range").onclick=()=>{if(!current)return;$("range-start").value=0;$("range-end").value=current.frames-1;updateRange();};
  $("next-anomaly").onclick=()=>{if(!current)return;const p=getPart($("part").value);let found=-1;for(let i=frame+1;i<current.frames;i++){if(distance(p.raw[i],p.raw[i-1])>100||distance(p.repaired[i],p.repaired[i-1])>100){found=i;break;}const angle=q=>{let a=q[i].slice(3),b=q[i-1].slice(3);return Math.acos(Math.min(1,Math.abs(a.reduce((s,v,j)=>s+v*b[j],0))/Math.hypot(...a)/Math.hypot(...b)))*360/Math.PI;};if(angle(p.raw)>10||angle(p.repaired)>10){found=i;break;}}video.pause();setFrame(found<0?0:found);};
  canvas.onclick=e=>{if(!current)return;const r=canvas.getBoundingClientRect();const f=range[0]+(e.clientX-r.left-72)/(r.width-92)*(range[1]-range[0]);video.pause();setFrame(f);};
  addEventListener("resize",drawChart);document.addEventListener("keydown",e=>{if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName))return;if(e.key==="ArrowRight"){video.pause();setFrame(frame+1);e.preventDefault();}if(e.key==="ArrowLeft"){video.pause();setFrame(frame-1);e.preventDefault();}});
  renderList();const match=location.hash.match(/episode=(\d+)/);load(match?Number(match[1]):manifest.episodes[0]?.episode);
})();
