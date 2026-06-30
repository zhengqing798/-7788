/**
 * 电影推荐系统 v2.0 - 前端交互
 */

// ============================================================
// 导航栏搜索自动补全
// ============================================================
(function() {
    const searchInput = document.getElementById('navSearch');
    const dropdown = document.getElementById('searchDropdown');
    let debounceTimer = null;

    if (!searchInput || !dropdown) return;

    searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        clearTimeout(debounceTimer);

        if (query.length < 1) {
            dropdown.classList.remove('active');
            return;
        }

        debounceTimer = setTimeout(() => {
            fetch(`/api/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    if (!data.results || data.results.length === 0) {
                        dropdown.innerHTML = '<div style="padding: 16px; color: #999; text-align: center;">无匹配结果</div>';
                    } else {
                        dropdown.innerHTML = data.results.slice(0, 8).map(m => `
                            <a href="/movie/${m.id}" class="search-dropdown-item">
                                ${m.poster_local_path
                                    ? `<img src="/static/${m.poster_local_path.replace(/\\/g, '/')}"
                                         alt="${m.title}"
                                         onerror="this.src='data:image/svg+xml,<svg xmlns=%27http://www.w3.org/2000/svg%27 width=%2736%27 height=%2750%27><rect fill=%27%23333%27 width=%2736%27 height=%2750%27/><text fill=%27%23666%27 x=%2718%27 y=%2728%27 text-anchor=%27middle%27 font-size=%2712%27>🎬</text></svg>'">`
                                    : '<div style="width:36px;height:50px;background:#333;border-radius:4px;display:flex;align-items:center;justify-content:center;">🎬</div>'}
                                <div class="item-info">
                                    <div class="item-title">${m.title}</div>
                                    <div class="item-meta">⭐ ${m.rating ? m.rating.toFixed(1) : 'N/A'} | ${(m.genre || '').slice(0, 20)}</div>
                                </div>
                            </a>
                        `).join('');
                    }
                    dropdown.classList.add('active');
                })
                .catch(() => {
                    dropdown.innerHTML = '<div style="padding: 16px; color: #e50914; text-align: center;">搜索失败</div>';
                    dropdown.classList.add('active');
                });
        }, 300);
    });

    // 回车搜索
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            const query = this.value.trim();
            if (query) {
                window.location.href = `/search?q=${encodeURIComponent(query)}`;
            }
        }
    });

    // 点击外部关闭
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.remove('active');
        }
    });
})();

// ============================================================
// 卡片入场动画
// ============================================================
(function() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .movie-card {
            animation: fadeInUp 0.4s ease forwards;
            animation-delay: calc(var(--card-index, 0) * 0.04s);
        }
    `;
    document.head.appendChild(style);

    document.querySelectorAll('.movie-card').forEach((card, index) => {
        card.style.setProperty('--card-index', index);
    });
})();

// ============================================================
// 平滑回到顶部
// ============================================================
(function() {
    const btn = document.createElement('button');
    btn.innerHTML = '↑';
    btn.style.cssText = `
        position: fixed; bottom: 30px; right: 30px; z-index: 999;
        width: 48px; height: 48px; border-radius: 50%;
        background: var(--accent); color: #fff; border: none;
        font-size: 24px; cursor: pointer; box-shadow: 0 4px 16px rgba(229,9,20,0.4);
        display: none; transition: all 0.3s ease;
    `;
    btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
    btn.addEventListener('mouseenter', () => btn.style.transform = 'scale(1.1)');
    btn.addEventListener('mouseleave', () => btn.style.transform = 'scale(1)');
    document.body.appendChild(btn);

    window.addEventListener('scroll', () => {
        btn.style.display = window.scrollY > 500 ? 'block' : 'none';
    });
})();
