// Fixed semantic payloads only. Failure never blocks a task operation.
function learningEvent(api, kind, control, target) {
  void api('/api/ai-assistant/learning/events', 'POST', {event_id:crypto.randomUUID(),kind,control,target}) .catch(()=>{});
}
function learningFeedback(parent, api, target) {
  const box=document.createElement('details'), summary=document.createElement('summary');
  summary.textContent='Feedback'; box.append(summary); let sequence=0;
  const verdictLabels={resolved:'Resolved',unresolved:'Unresolved',modify:'Needs changes'};
  const reason=document.createElement('select'); reason.setAttribute('aria-label','Feedback reason');
  for(const [key,label] of Object.entries({none:'Reason (optional)',wrong_operator:'Wrong operator',wrong_order:'Wrong order',wrong_input:'Wrong input',wrong_parameters:'Wrong parameters',missing_review:'Missing review',unclear:'Unclear explanation',execution_failure:'Execution failure',infrastructure:'Infrastructure problem',other:'Other'})){
    const option=document.createElement('option');option.value=key;option.textContent=label;reason.append(option);
  }
  box.append(reason); const status=document.createElement('span');status.setAttribute('role','status');
  for(const [verdict,label] of Object.entries({resolved:'Resolved',unresolved:'Unresolved',modify:'Needs changes'})){
    const button=document.createElement('button');button.type='button';button.textContent=label;
    button.onclick=async()=>{sequence++;const buttons=box.querySelectorAll('button');buttons.forEach(b=>b.disabled=true);
      try{const previous=await api('/api/ai-assistant/learning/feedback?'+new URLSearchParams(target));
        await api('/api/ai-assistant/learning/feedback','POST',{event_id:crypto.randomUUID(),expected_revision:previous?.revision||0,target,verdict,reason:reason.value});
        summary.textContent='Feedback · '+label;status.textContent='Assessment saved. Execution verification is recorded separately.';
      }catch(e){status.textContent=e.message;}finally{buttons.forEach(b=>b.disabled=false);}
    };box.append(button);
  }
  box.ontoggle=async()=>{if(!box.open)return;const expected=++sequence;
    try{const value=await api('/api/ai-assistant/learning/feedback?'+new URLSearchParams(target));
      if(expected===sequence)summary.textContent='Feedback · '+(verdictLabels[value.verdict]||'No feedback');
    }catch(e){if(expected===sequence)status.textContent=e.message;}
  };
  box.append(status); parent.append(box);return box;
}
