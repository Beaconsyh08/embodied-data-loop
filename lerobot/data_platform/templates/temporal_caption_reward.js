// Read-only reward evidence. Never convert a model score into a human review decision.
const rewardColors = ['#334155','#fb7185','#fbbf24','#34d399'];
function rewardRange(start,end,fps) { return `${(start/fps).toFixed(2)}–${(end/fps).toFixed(2)} s`; }
function clearReward() {
    for(const id of ['reward-review','reward-summary','reward-details','reward-overlay'])$(id).hidden=true;
    for(const id of ['reward-tracks','reward-values','reward-conflicts','reward-raw','disagreement-bands'])$(id).replaceChildren();
    for(const id of ['reward-terminal','reward-overall-score','reward-merged-score','reward-merged-source','reward-score-time'])$(id).textContent='';
    $('reward-score-conflict').hidden=true;
}
function rewardSeek(frame) {
    if(!artifact)return;
    $('loop').checked=false;
    $('video').currentTime=Math.max(0,Math.min(artifact.frame_count-1,frame))/artifact.fps;
}
function renderReward(data) {
    clearReward();
    const reward=data.reward;if(!reward)return;
    for(const id of ['reward-review','reward-summary','reward-details','reward-overlay'])$(id).hidden=false;
    $('reward-overall-score').textContent=reward.terminal.score;
    $('reward-terminal').textContent=reward.terminal.rationale;
    const windows=document.createElement('details');windows.className='reward-windows';windows.open=reward.windows.length<=6;
    windows.append(text('summary',`Windows (${reward.windows.length})`));
    const tracks=[{label:'Merged Reward',scores:reward.merged_scores},...reward.windows.map((w,i)=>({label:`Window ${i+1}`,scores:w.scores,index:i}))];
    for(const track of tracks) {
        const row=text('div','','timeline-row reward-track'),label=text('span',track.label,'timeline-label'),canvas=document.createElement('canvas');
        const start=track.scores.findIndex(Boolean);let end=track.scores.length;
        while(end>0&&!track.scores[end-1])end--;
        const range=start<0?'No score coverage':rewardRange(start,end,data.fps);
        label.append(text('small',range));
        if(track.index!==undefined)row.dataset.window=track.index;
        canvas.width=Math.min(data.frame_count,4096);canvas.height=22;canvas.tabIndex=0;canvas.setAttribute('role','slider');
        canvas.setAttribute('aria-label',`${track.label}, ${range}: seek video`);canvas.setAttribute('aria-valuemin','0');canvas.setAttribute('aria-valuemax',String(data.frame_count-1));
        canvas.setAttribute('aria-valuenow','0');
        const context=canvas.getContext('2d'),scale=canvas.width/data.frame_count;
        for(let start=0;start<track.scores.length;) {
            let end=start+1;while(end<track.scores.length&&track.scores[end]===track.scores[start])end++;
            context.fillStyle=rewardColors[track.scores[start]];context.fillRect(start*scale,0,(end-start)*scale,22);start=end;
        }
        canvas.onclick=event=>{const rect=canvas.getBoundingClientRect();rewardSeek(Math.floor((event.clientX-rect.left)/rect.width*data.frame_count));};
        canvas.onkeydown=event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();rewardSeek(Math.floor($('video').currentTime*data.fps)+(event.key==='ArrowRight'?1:-1));};
        const surface=text('div','','reward-track-surface');surface.append(canvas);row.append(label,surface);
        (track.index===undefined?$('reward-tracks'):windows).append(row);
    }
    $('reward-tracks').append(windows);
    if(!reward.disagreement_intervals.length)$('reward-conflicts').append(text('span','No overlapping-window score disagreements.'));
    for(const span of reward.disagreement_intervals) {
        const range=rewardRange(span.start_frame,span.end_frame,data.fps);
        const button=text('button',`⚠ ${range}`,'dp-btn reward-conflict');
        button.title='Review disagreeing reward windows';button.onclick=()=>rewardSeek(span.start_frame);$('reward-conflicts').append(button);
        const band=text('span','','disagreement-band');band.style.left=`${span.start_frame/data.frame_count*100}%`;
        band.style.width=`${(span.end_frame-span.start_frame)/data.frame_count*100}%`;$('disagreement-bands').append(band);
    }
    reward.windows.forEach((window,i)=>{const detail=document.createElement('details');detail.append(text('summary',`Window ${i+1} · raw model output`),text('pre',window.output));$('reward-raw').append(detail);});
    $('reward-raw').append(text('p',reward.alignment,'note'));
}
function updateReward(frame) {
    const reward=artifact?.reward;if(!reward)return;
    const scores=reward.windows.map(w=>w.scores[frame]);
    const conflict=new Set(scores.filter(Boolean)).size>1;
    const merged=reward.merged_scores[frame],source=reward.merged_source_window[frame];
    $('reward-overlay').textContent=`Action Reward ${merged}/3 · Window ${source+1}${conflict?' · Windows disagree':''}`;
    $('reward-overlay').classList.toggle('reward-disagreement',conflict);
    $('reward-values').replaceChildren();
    $('reward-score-time').textContent=`· ${ (frame/artifact.fps).toFixed(2) } s`;
    $('reward-merged-score').textContent=`${merged} / 3`;
    $('reward-merged-source').textContent=`from Window ${source+1}`;
    $('reward-score-conflict').hidden=!conflict;
    scores.forEach((score,i)=>{
        if(!score)return;
        const item=text('div','','reward-window-score');item.classList.toggle('is-source',i===source);
        item.style.setProperty('--score-color',rewardColors[score]);
        const value=text('div','','reward-window-value');value.append(text('strong',score),text('span','/ 3'));
        item.append(text('div',`Window ${i+1}`,'reward-window-name'),value,
            text('small',i===source?'Used by Merged':'Overlapping window'));
        $('reward-values').append(item);
    });
    for(const row of $('reward-tracks').querySelectorAll('[data-window]'))row.classList.toggle('reward-source',Number(row.dataset.window)===source);
    for(const canvas of $('reward-tracks').querySelectorAll('canvas'))canvas.setAttribute('aria-valuenow',String(frame));
}
