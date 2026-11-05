// Navigation is explicit; a late catalog response must never change the selected dataset.
const datasetPicker = document.getElementById('caption-datasets');
const datasetSearch = document.getElementById('caption-datasets-search');
const datasetList = document.getElementById('caption-datasets-list');
const datasetStatus = document.getElementById('caption-datasets-status');
const datasetRefresh = document.getElementById('caption-datasets-refresh');
const currentDataset = {{ caption_dataset_id|tojson }};
let datasetRows = [], datasetLoaded = false, datasetLoading = false, datasetError = '';
function renderDatasetPicker() {
    const query = datasetSearch.value.trim().toLocaleLowerCase();
    const rows = datasetRows.filter(row => `${row.name} ${row.dataset_key} ${row.node}`.toLocaleLowerCase().includes(query));
    datasetList.replaceChildren();
    for (const row of rows) {
        const link = document.createElement('a');
        link.href = row.review_url;
        // Do not carry an episode/run across dataset identities. The saved method preference
        // is restored by the destination only when that method has results there.
        if (row.id === currentDataset) link.setAttribute('aria-current', 'page');
        const name = document.createElement('strong'); name.textContent = row.name;
        const info = document.createElement('span');
        const sources = [];
        if (row.saved_runs) sources.push(`${row.saved_runs} saved runs on platform`);
        if (row.agent_runs) sources.push(`${row.agent_runs} results on Agent (last scan)${row.agent_online ? '' : ' · Agent offline'}`);
        info.textContent = `${row.node} · ${sources.join(' · ')}${row.id === currentDataset ? ' · Current' : ''}`;
        const key = document.createElement('small'); key.textContent = row.dataset_key;
        link.append(name, info, key); datasetList.append(link);
    }
    datasetStatus.textContent = datasetError || (datasetLoading ? 'Loading annotated datasets…' :
        rows.length ? `${rows.length} / ${datasetRows.length} datasets · platform and Agent results` :
        query ? 'No matching annotated datasets.' : 'No annotation results available to this account.');
}
async function loadDatasetPicker() {
    if (datasetLoading) return;
    datasetLoading = true; datasetRefresh.disabled = true; datasetError = ''; renderDatasetPicker();
    try {
        const response = await fetch('/api/temporal-caption/datasets');
        if (!response.ok) throw new Error(`Unable to load datasets (HTTP ${response.status}). Use Refresh to retry.`);
        const data = await response.json(); datasetRows = data.datasets; datasetLoaded = true;
        datasetError = (data.warnings || []).join(' ');
    } catch (error) { datasetError = error.message; }
    finally { datasetLoading = false; datasetRefresh.disabled = false; renderDatasetPicker(); }
}
datasetPicker.addEventListener('toggle', () => {
    if (!datasetPicker.open) return;
    datasetSearch.focus(); if (!datasetLoaded) loadDatasetPicker();
});
datasetSearch.addEventListener('input', renderDatasetPicker);
datasetRefresh.addEventListener('click', loadDatasetPicker);
datasetPicker.addEventListener('keydown', event => {
    if (event.key === 'Escape') { datasetPicker.open = false; datasetPicker.querySelector('summary').focus(); }
});
document.addEventListener('click', event => { if (!datasetPicker.contains(event.target)) datasetPicker.open = false; });
