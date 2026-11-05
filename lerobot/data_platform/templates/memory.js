(() => {
'use strict';
const $ = id => document.getElementById(id), base = `/api/control/locations/${encodeURIComponent(MEMORY_CONFIG.location_id)}/memory`;
let auditBatch=null;
let rows=[], jobs=[], prepared=[], pilot=null, review=null, run=null, stopped=false, playback=null, exportRequest=null;
const cacheKey=`memory-pilot:${MEMORY_CONFIG.location_id}`;
const pendingKey=`memory-submit:${MEMORY_CONFIG.location_id}`;
try { pilot=JSON.parse(sessionStorage.getItem(cacheKey)||'null'); } catch (_) {}
const status=(message,error=false)=>{ $('status').textContent=message; $('status').classList.toggle('error',error); };
async function api(path='',body,delivery){
 const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json',...(delivery?{'Idempotency-Key':delivery}:{})},...(body===undefined?{}:{body:JSON.stringify(body)})});
 const value=await response.json();if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);return value;
}
const safe=fn=>async()=>{try{await fn();}catch(error){status(error.message,true);}};
const option=(value,label)=>{const node=document.createElement('option');node.value=value;node.textContent=label;return node;};
function showPilot(){ $('selection').textContent=pilot?JSON.stringify({episodes:pilot.selected.map(p=>p.episode_index),splits:pilot.counts,shortfall:pilot.shortfall,requests:pilot.selected.reduce((sum,p)=>sum+p.expected_requests,0)},null,2):''; $('submit').disabled=!pilot||!pilot.selected.length||!MEMORY_CONFIG.can_edit; }
const reviewRuns=()=>jobs.filter(j=>j.operation==='memory.annotate'&&j.status==='done').flatMap(j=>j.result?.summary?.episodes?j.result.summary.episodes.filter(e=>e.has_review).map(e=>({...j,job_id:e.run,episode_index:e.episode_index,batch_job_id:j.job_id})):[j]);
async function refresh(){
 const state=await api();jobs=state.runs; $('audit').disabled=!state.can_submit;
 if(state.unavailable_reason)status(state.unavailable_reason,true);
 const chosen=$('runs').value; $('runs').replaceChildren(...reviewRuns().map(j=>option(j.job_id,`${j.episode_index!==undefined?'Episode '+j.episode_index+' · ':''}${j.job_id.slice(0,8)} · ${j.created_at}`))); if([...$('runs').options].some(o=>o.value===chosen))$('runs').value=chosen;
 const exports=jobs.filter(j=>j.operation==='memory.export');if(exports.length)$('export-result').textContent=JSON.stringify(exports.map(j=>({job:j.job_id,status:j.status,result:j.result})),null,2);
}
async function initialize(){
 const response=await fetch(`/api/control/locations/${encodeURIComponent(MEMORY_CONFIG.location_id)}/temporal-caption`);
 const value=await response.json();if(!response.ok)throw new Error(value.error||'Unable to read FLORA runs');
 const location=await api();
 const count=location.total_episodes;
 if(!Number.isInteger(count))throw new Error('Dataset episode inventory unavailable; refresh its registration');
 const runs=value.runs.filter(r=>(r.memory_candidate||r.scheme?.id==='flora')&&r.variant==='caption');
 rows=Array.from({length:count},(_,episode)=>({episode,runs:runs.filter(r=>r.episode_index===episode)}));
 $('episodes').replaceChildren();
 for(const row of rows){
  const tr=document.createElement('tr');const episode=document.createElement('td');episode.textContent=row.episode;
  const choice=document.createElement('td');row.select=document.createElement('select');row.select.setAttribute('aria-label',`FLORA run episode ${row.episode}`);row.select.replaceChildren(...row.runs.map(r=>option(r.run,r.run)));choice.append(row.select);
  const group=document.createElement('td');row.group=document.createElement('input');row.group.placeholder='Original recording ID';row.group.setAttribute('aria-label',`Recording group episode ${row.episode}`);group.append(row.group);
  row.status=document.createElement('td');row.status.textContent=row.runs.length?'Not checked':'No complete FLORA run';tr.append(episode,choice,group,row.status);$('episodes').append(tr);
 }
 await refresh();showPilot();const requested=new URLSearchParams(window.location.search).get('run');if(requested&&[...$('runs').options].some(o=>o.value===requested)){$('runs').value=requested;await open();}if(MEMORY_CONFIG.can_edit)status('Ready for source audit. No model requests have been sent.');
}
$('audit').onclick=safe(async()=>{
 if(!$('task').value.trim())throw new Error('Enter the explicit total task');prepared=[];auditBatch=crypto.randomUUID();$('freeze').disabled=true;$('audit').disabled=true;
 try{for(const row of rows){if(stopped)return;try{
  if(row.select.value&&!row.group.value.trim())throw new Error('Recording group required');
  const result=await api('/prepare',{episode_index:row.episode,flora_run:row.select.value,task:$('task').value,lineage_group:row.group.value.trim(),audit_batch:auditBatch});
  prepared.push(result);row.status.textContent=`${result.pin.checkpoints.length} checkpoints · ${result.pin.expected_requests} model requests`;
 }catch(error){row.status.textContent=error.message;}}}finally{$('audit').disabled=false;}
 $('freeze').disabled=false;status(`${prepared.length}/${rows.length} episodes passed source audit; no model requests sent.`);
});
$('freeze').onclick=safe(async()=>{pilot=await api('/pilot',{request_ids:prepared.map(p=>p.request_id),seed:42,audit_batch:auditBatch});pilot.requests=prepared.filter(p=>pilot.selected.some(s=>s.episode_index===p.pin.episode_index)).map(p=>p.request_id);sessionStorage.setItem(cacheKey,JSON.stringify(pilot));showPilot();status('Selection and whole-recording splits frozen. Review the counts before submission.');});
$('submit').onclick=safe(async()=>{
 let pending;try{pending=JSON.parse(sessionStorage.getItem(pendingKey)||'null');}catch(_){}
 if(pending?.ids)throw new Error('An older per-episode submission is pending; check Jobs before starting a dataset batch');
 if(!pending||pending.pilot!==pilot.digest){
  if(!pilot.requests?.length)throw new Error('Audit and freeze this pilot before submission');
  const staged=await api('/batch',{request_ids:pilot.requests,pilot_id:pilot.digest});
  pending={pilot:pilot.digest,request:staged.request_id,key:crypto.randomUUID(),episodes:staged.episodes,parallel:staged.max_parallel_episodes};
  sessionStorage.setItem(pendingKey,JSON.stringify(pending));
 }
 $('submit').disabled=true;
 try{await api('/jobs',{request_id:pending.request},pending.key);sessionStorage.removeItem(pendingKey);await refresh();status(`Dataset Job submitted for ${pending.episodes} episodes; up to ${pending.parallel} episodes run concurrently. Follow progress in Jobs.`);}finally{$('submit').disabled=false;}
});
function render(){
 clearInterval(playback);$('playback').hidden=true;
 const state=review.state,index=Number($('checkpoint').value),entry=state.entries[index],point=state.record.checkpoints.find(p=>p.frame===entry.frame);
 $('revision').textContent=`Revision ${review.revision} · ${state.record.review.status} · episode ${state.pin.episode_index}`;
 $('issues').textContent=`Frame ${entry.frame} · ${entry.state}\n${entry.issues.join('\n')}`;
 const frameUrl=frame=>`${base}/reviews/${encodeURIComponent(run)}/frames/${frame}.png?cutoff=${entry.frame}`;
 $('current').src=frameUrl(entry.frame);$('history').replaceChildren();
 for(const frame of entry.history_frames){const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');image.src=frameUrl(frame);image.alt=`Past frame ${frame}`;caption.textContent=`Frame ${frame}`;figure.append(image,caption);$('history').append(figure);}
 $('previous').textContent=state.record.checkpoints.filter(p=>p.frame<entry.frame).at(-1)?.memory||'No events have been confirmed yet.';
 $('ledger').textContent=JSON.stringify(state.record.facts.filter(f=>f.confirmed_at_frame<=entry.frame),null,2);
 $('memory').value=point?.memory||'';$('subtask').value=point?.subtask||'';$('facts').value=JSON.stringify(state.record.facts.filter(f=>f.confirmed_at_frame===entry.frame),null,2);$('point').value=JSON.stringify(point||{},null,2);$('reason').value='';$('flora-text').textContent='';
 for(const id of ['source-confirmed','goal-confirmed','causal-confirmed'])$(id).checked=false;
 $('editor').disabled=!MEMORY_CONFIG.can_edit;$('approval').disabled=!MEMORY_CONFIG.can_edit;
}
async function open(){run=$('runs').value;if(!run)throw new Error('No completed memory draft');review=await api(`/reviews/${encodeURIComponent(run)}`);$('review').hidden=false;$('checkpoint').replaceChildren(...review.state.entries.map((e,i)=>option(i,`Frame ${e.frame} · ${e.state}`)));render();}
const entry=()=>review.state.entries[Number($('checkpoint').value)];
async function change(action,body={}){const selected=$('checkpoint').value;review=await api(`/reviews/${encodeURIComponent(run)}/${action}`,{expected_revision:review.revision,...body});$('checkpoint').value=selected;render();status(`Saved revision ${review.revision}`);}
$('open').onclick=safe(open);$('refresh').onclick=safe(refresh);$('checkpoint').onchange=render;
$('save').onclick=safe(async()=>{const point=JSON.parse($('point').value);point.frame=entry().frame;point.memory=$('memory').value;point.subtask=$('subtask').value;await change('edit',{frame:entry().frame,facts:JSON.parse($('facts').value),checkpoint:point,reason:$('reason').value});});
$('pending').onclick=safe(()=>change('needs_review',{frame:entry().frame}));
$('add').onclick=safe(()=>change('checkpoint',{frame:Number($('extra').value)}));
$('regenerate').onclick=safe(async()=>{const result=await api(`/reviews/${encodeURIComponent(run)}/regenerate`,{expected_revision:review.revision});await api('/jobs',{request_id:result.request_id},result.request_id);status('Regeneration submitted. Previous revisions remain available; see Jobs.');await refresh();});
function confirmations(){return {source_confirmed:$('source-confirmed').checked,goal_confirmed:$('goal-confirmed').checked,causal_evidence_confirmed:$('causal-confirmed').checked};}
$('approve').onclick=safe(()=>change('approve',confirmations()));$('independent').onclick=safe(()=>change('approve',{...confirmations(),independent:true,adjudicated:true}));
$('flora').onclick=safe(async()=>{$('flora-text').textContent=JSON.stringify(await api(`/reviews/${encodeURIComponent(run)}/flora?cutoff=${entry().frame}`),null,2);});
$('play').onclick=()=>{clearInterval(playback);const urls=[...$('history').querySelectorAll('img')].map(im=>im.src);urls.push($('current').src);let i=0;$('playback').hidden=false;$('playback').src=urls[0];playback=setInterval(()=>{i++;if(i>=urls.length){clearInterval(playback);return;}$('playback').src=urls[i];},250);};
$('export').onclick=safe(async()=>{if(!pilot)throw new Error('Freeze the pilot first');await refresh();const selected=[];for(const pin of pilot.selected){let found=false;for(const job of reviewRuns()){const value=await api(`/reviews/${encodeURIComponent(job.job_id)}`);if(value.state.pin.episode_index===pin.episode_index){selected.push(job.job_id);found=true;break;}}if(!found)throw new Error(`Episode ${pin.episode_index} has no completed draft`);}const result=await api('/export',{runs:selected,pilot_id:pilot.digest});exportRequest=result.request_id;$('export-submit').hidden=false;$('export-result').textContent=`Prepared reviewed export for ${selected.length} episodes. No export Job has run yet.`;});
$('export-submit').onclick=safe(async()=>{await api('/jobs',{request_id:exportRequest},exportRequest);$('export-submit').hidden=true;status('Export submitted; follow Jobs and validate the package with mem_tools.');});
if(!MEMORY_CONFIG.can_edit)for(const id of ['audit','freeze','submit','export','export-submit'])$(id).disabled=true;
window.addEventListener('pagehide',()=>{stopped=true;clearInterval(playback);});
safe(initialize)();
})();
