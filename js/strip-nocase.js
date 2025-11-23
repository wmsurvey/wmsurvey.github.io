document.addEventListener('DOMContentLoaded', () => {
  for (const node of document.querySelectorAll('.md-content')) {
    // 只替换文本中的 "]{.nocase}" / "]{.nocase} " 之类
    node.innerHTML = node.innerHTML.replace(/\]\{\.nocase\}\s*/g, '');
  }
});
