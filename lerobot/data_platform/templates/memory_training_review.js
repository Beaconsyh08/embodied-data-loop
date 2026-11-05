(() => {
'use strict';
const $=id=>document.getElementById(id);
const base=`/api/control/locations/${encodeURIComponent(MEMORY_REVIEW_LOCATION)}`;
let index=[], episode=null, request=0, requestedEpisode=new URLSearchParams(location.search).get('episode');
const error=message=>{$('status').textContent=message;};
function clearReview(){for(const id of ['position','coverage','task','previous-memory','memory','subtask','issues','facts','cameras'])$(id).textContent='';$('points').replaceChildren();}
async function get(path){const response=await fetch(base+path);const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;}
function updateOptions(){
 const old=$('episodes').value, filter=$('filter').value;
 $('episodes').replaceChildren(...index.filter(e=>filter==='all'||(filter==='complete'?e.complete:!e.complete)).map(e=>{
  const o=document.createElement('option');o.value=e.episode_index;o.textContent=`Episode ${e.episode_index} · ${e.complete?'complete draft':'needs review'} · ${e.accepted}/${e.planned}`;return o;
 }));
 if(requestedEpisode!==null&&[...$('episodes').options].some(o=>o.value===requestedEpisode))$('episodes').value=requestedEpisode;
 else if([...$('episodes').options].some(o=>o.value===old))$('episodes').value=old;
 requestedEpisode=null;
 if($('episodes').value!=='')openEpisode();else {request++;episode=null;$('video').removeAttribute('src');$('video').load();clearReview();error('No episodes match this filter.');}
}
function render(frame){
 if(!episode)return;
 const pending=episode.stopped_at!==null&&frame>=episode.stopped_at;
 let i=-1;for(let n=0;n<episode.checkpoints.length;n++){if(episode.checkpoints[n].frame>frame)break;i=n;}
 const current=i>=0?episode.checkpoints[i]:null;
 $('position').textContent=`Frame ${frame} · ${(frame/episode.fps).toFixed(2)} s${pending?' · needs review':''}`;
 $('previous-memory').textContent=pending&&current?current.memory:i>0?episode.checkpoints[i-1].memory:'No events have been confirmed yet.';
 $('memory').textContent=pending?'No accepted memory at or after the stopped checkpoint.':current?.memory||'No label at this frame.';
 $('subtask').textContent=pending?'No accepted subtask at or after the stopped checkpoint.':current?.subtask||'No label at this frame.';
 const issue=episode.issues;
 $('issues').textContent=issue?(issue.revisions||[]).map(r=>`Revision ${r.revision}:\n${(r.issues||[]).join('\n')}`).join('\n\n') || (issue.issues||[]).join('\n') || 'Review required; no detail was recorded.':'No stopped checkpoint in this episode.';
 $('facts').textContent=JSON.stringify(episode.facts.filter(f=>f.confirmed_at_frame<=frame),null,2);
 for(const b of $('points').children)b.classList.toggle('active',pending?b.dataset.pending==='true':Number(b.dataset.index)===i);
}
async function openEpisode(){
 const id=Number($('episodes').value), serial=++request;error('Loading episode…');episode=null;$('video').removeAttribute('src');$('video').load();clearReview();
 try{
  const e=await get(`/memory/training-review/${id}`);if(serial!==request)return;episode=e;
  $('coverage').textContent=`${e.accepted}/${e.planned} accepted checkpoints · ${e.complete?'complete model draft':`stopped at frame ${e.stopped_at}` } · ${e.split}`;
  $('task').textContent=e.task;
  const names={image:'Main',head:'Main',scene:'Main',left_wrist_image:'Left wrist',left_wrist:'Left wrist',right_wrist_image:'Right wrist',right_wrist:'Right wrist'};
  $('cameras').textContent=e.video_cameras?.length?e.video_cameras.map(c=>names[c]||c).join(' · '):'Camera order unavailable.';
  for(const [i,p] of e.checkpoints.entries()){const b=document.createElement('button');b.textContent=`${(p.frame/e.fps).toFixed(1)}s`;b.dataset.index=i;b.onclick=()=>{$('video').currentTime=p.frame/e.fps;render(p.frame);};$('points').append(b);}
  if(e.stopped_at!==null){const b=document.createElement('button');b.textContent=`⚠ ${(e.stopped_at/e.fps).toFixed(1)}s`;b.className='pending';b.dataset.pending='true';b.onclick=()=>{$('video').currentTime=e.stopped_at/e.fps;render(e.stopped_at);};$('points').append(b);}
  $('video').src=`${base}/temporal-caption/${encodeURIComponent(e.caption_run)}/caption/video`;
  error('Read-only model draft; video and labels are bound to the frozen source.');render(0);
 }catch(err){if(serial===request)error(err.message);}
}
$('video').addEventListener('timeupdate',()=>{if(episode)render(Math.min(episode.frame_count-1,Math.floor($('video').currentTime*episode.fps)));});
$('video').addEventListener('error',()=>error('Video unavailable. The label remains read-only; retry when the registered source video is available.'));
$('episodes').onchange=openEpisode;$('filter').onchange=updateOptions;
for(const [id,step] of [['previous',-1],['next',1]])$(id).onclick=()=>{const select=$('episodes');select.selectedIndex=Math.max(0,Math.min(select.options.length-1,select.selectedIndex+step));openEpisode();};
(async()=>{try{const data=await get('/memory/training-review');index=data.episodes;$('summary').textContent=`${data.requested_episodes} episodes · ${data.accepted_checkpoints} model-checked checkpoints · ${data.complete_episodes} complete drafts · ${data.requested_episodes-data.complete_episodes} stopped for review`;updateOptions();}catch(err){error(err.message);}})();
})();
