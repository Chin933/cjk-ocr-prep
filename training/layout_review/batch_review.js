// Review state is separate from immutable predictions and human annotations.
(() => {
  const batchId = location.pathname.split('/').pop() + ':' + pages[0].fingerprint;
  const key = 'layout-batch-review:' + batchId;
  const states = new Set(['unreviewed', 'ok', 'issue']);
  const names = new Set(pages.map(p => p.name));
  let records = Object.create(null), loaded = false;

  function validate(data) {
    if (data.version !== 1 || data.batchId !== batchId || !data.pages || typeof data.pages !== 'object')
      throw new Error('这不是当前批次的反馈文件。');
    const clean = Object.create(null);
    for (const [name, value] of Object.entries(data.pages)) {
      if (!names.has(name) || !value || !states.has(value.status) || typeof value.note !== 'string')
        throw new Error('反馈内容格式不完整，未导入。');
      clean[name] = {status: value.status, note: value.note, updatedAt: value.updatedAt || null};
    }
    return clean;
  }
  function snapshot() {return {version: 1, batchId, pages: records};}
  function persist() {
    try {
      localStorage.setItem(key, JSON.stringify(snapshot()));
      $('saveState').textContent = '已自动保存到本浏览器 · 建议导出备份';
    } catch {
      $('saveState').textContent = '浏览器保存失败，请立即导出反馈';
    }
  }
  try {
    const saved = localStorage.getItem(key);
    if (saved) {records = validate(JSON.parse(saved)); loaded = true;}
  } catch {
    $('saveState').textContent = '未能读取浏览器反馈；可从导出文件导入';
  }
  function updateCount() {
    const reviewed = Object.values(records).filter(r => r.status !== 'unreviewed').length;
    const issues = Object.values(records).filter(r => r.status === 'issue').length;
    $('reviewCount').textContent = `已检查 ${reviewed}/${pages.length} · 有问题 ${issues}`;
    [...$('pageSelect').options].forEach((option, index) => {
      const p = pages[index], s = records[p.name]?.status || 'unreviewed';
      option.textContent = `${index+1} · ${p.name.replace('.jpg','')} ${p.human.length?'[训练参考]':''} ${p.diagnostic?'[空结果]':''} ${s==='ok'?'✓':s==='issue'?'⚑':''}`;
    });
  }
  function loadPage() {
    const record = records[pages[current].name];
    $('reviewState').value = record?.status || 'unreviewed';
    $('reviewText').value = record?.note || '';
    $('previousPage').disabled = current === 0;
    $('nextPage').disabled = current === pages.length-1;
    updateCount();
  }
  function savePage() {
    records[pages[current].name] = {status: $('reviewState').value,
      note: $('reviewText').value, updatedAt: new Date().toISOString()};
    persist(); updateCount();
  }
  $('reviewText').oninput = () => {
    if ($('reviewText').value.trim() && $('reviewState').value === 'unreviewed') $('reviewState').value = 'issue';
    savePage();
  };
  $('reviewState').onchange = savePage;
  const selectPage = $('pageSelect').onchange;
  $('pageSelect').onchange = () => {
    selectPage(); loadPage();
    const url = new URL(location.href); url.searchParams.set('page', current);
    history.replaceState(null, '', url);
  };
  function go(index) {
    $('pageSelect').value = String(Math.max(0, Math.min(pages.length-1, index)));
    $('pageSelect').onchange();
  }
  function findNext(predicate) {
    for (let i=1;i<=pages.length;i++) {
      const next=(current+i)%pages.length;
      if (predicate(pages[next])) {go(next);return;}
    }
    $('saveState').textContent = '没有符合条件的页面';
  }
  $('previousPage').onclick = () => go(current-1);
  $('nextPage').onclick = () => go(current+1);
  $('nextUnreviewed').onclick = () => findNext(p => !records[p.name] || records[p.name].status === 'unreviewed');
  $('nextEmpty').onclick = () => findNext(p => Boolean(p.diagnostic));
  $('exportReview').onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(),null,2)],{type:'application/json'}));
    const a = document.createElement('a'); a.href=url;a.download='notes_100.feedback.json';a.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  };
  $('importReview').onclick = () => $('importFile').click();
  $('importFile').onchange = async () => {
    const file=$('importFile').files[0];if (!file)return;
    try {
      const incoming=validate(JSON.parse(await file.text()));
      const overlap=Object.keys(incoming).filter(name => records[name]);
      if (overlap.length && !confirm(`将替换 ${overlap.length} 页已有的本地检查记录，是否继续？`)) return;
      records=Object.assign(Object.create(null),records,incoming);persist();loadPage();
    } catch (error) {$('saveState').textContent=error.message;}
    finally {$('importFile').value='';}
  };
  loadPage();
  if (loaded) $('saveState').textContent='已恢复本浏览器保存的检查记录';
})();
