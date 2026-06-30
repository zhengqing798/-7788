"""
豆瓣影人数据爬虫模块
- 按姓名搜索影人 → 获取影人页面 → 解析详情 → 下载头像
- 处理防盗链：图片请求带 Referer + 浏览器 UA
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Windows 终端 UTF-8 编码，避免 emoji 等字符报错
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ============================================================
# 配置
# ============================================================

# 豆瓣 Cookie（登录态 + personage 页面）
DOUBAN_COOKIE = (
    'bid=SlJWO_1i1zs; '
    'ck=-aKW; '
    'dbcl2="295727633:9giEOdx27zw"; '
    'frodotk_db="bae904c701e3493230bb0e5505a9db96"; '
    'll="118201"; '
    'push_doumail_num=0; '
    'push_noty_num=0; '
    '_vwo_uuid_v2=DB701F7847911E41D24D86CC09E875651|ddcc09ff5e061aad38521f1e7d9639b7; '
    'ap_v=0,6.0; '
    '__utma=30149280.965526527.1782112451.1782203946.1782347835.4; '
    '__utmb=30149280.2.10.1782347835; '
    '__utmc=30149280; '
    '__utmv=30149280.29572; '
    '__utmz=30149280.1782347835.4.4.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; '
    '_pk_id.100001.8cb4=8220a74c02256fb2.1782088954.; '
    '_pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1782347833%2C%22https%3A%2F%2Fcn.bing.com%2F%22%5D; '
    '_pk_ses.100001.8cb4=1'
)

# 浏览器 UA
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/125.0.0.0 Safari/537.36'
)

# 请求头模板（注意：不要加 Accept-Encoding / Upgrade-Insecure-Requests，会触发豆瓣反爬）
HEADERS = {
    'User-Agent': USER_AGENT,
    'Cookie': DOUBAN_COOKIE,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
}

# 图片请求专用头（带 Referer 防防盗链，doubanio.com CDN 需要豆瓣域名的 referer）
IMAGE_HEADERS = {
    'User-Agent': USER_AGENT,
    'Referer': 'https://www.douban.com/',
    'Origin': 'https://www.douban.com',
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 路径配置
BASE_DIR = Path(__file__).parent
PERSONS_DATA_FILE = BASE_DIR / 'persons_data.json'
AVATAR_DIR = BASE_DIR / 'static' / 'persons'
MOVIE_DATA_FILE = BASE_DIR / 'movies_data_temp(1).json'

# 请求间隔（秒），加随机抖动避免被封
MIN_DELAY = 0.3
MAX_DELAY = 1.0

# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def random_delay():
    """随机延迟，模拟人类浏览行为"""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def clean_text(text: str) -> str:
    """清理文本：去首尾空白、压缩多余换行"""
    if not text:
        return ''
    return re.sub(r'\n\s*\n', '\n', text.strip())

def load_persons_cache() -> dict:
    """加载已爬取的影人数据"""
    if PERSONS_DATA_FILE.exists():
        with open(PERSONS_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_persons_cache(data: dict):
    """保存影人数据到缓存文件"""
    ensure_dir(PERSONS_DATA_FILE.parent)
    with open(PERSONS_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# 搜索影人 ID
# ============================================================

def search_celebrity(name: str, known_movies: list[str] | None = None) -> str | None:
    """
    在豆瓣搜索影人，返回影人 ID（celebrity id）
    优先使用影人搜索，失败时回退到电影页面提取影人链接
    - name: 影人姓名
    - known_movies: 该影人参演的电影 ID 列表（用于兜底搜索）
    """
    from urllib.parse import quote

    celeb_id = _search_celebrity_direct(name)
    if celeb_id:
        return celeb_id

    # 兜底：从已知电影页面中提取影人链接
    if known_movies:
        celeb_id = _find_celebrity_from_movies(name, known_movies[:5])
        if celeb_id:
            return celeb_id

    return None


def _search_celebrity_direct(name: str, retries: int = 3) -> str | None:
    """
    直接搜索豆瓣影人
    """
    from urllib.parse import quote
    search_url = f'https://movie.douban.com/celebrities/search?search_text={quote(name)}'

    for attempt in range(retries):
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return None

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 搜索结果结构: div.result > a.nbg[href*="/celebrity/"]
            for result in soup.select('.result'):
                nbg_link = result.select_one('a.nbg[href*="/celebrity/"]')
                if not nbg_link:
                    continue
                href = nbg_link.get('href', '')
                m = re.search(r'/celebrity/(\d+)/?', href)
                if not m:
                    continue
                celeb_id = m.group(1)

                # 用 title 属性（最可靠）和 h3 文本一起校验
                title = nbg_link.get('title', '')
                content_link = result.select_one('.content h3 a') or result.select_one('h3 a')
                display_text = content_link.get_text(strip=True) if content_link else ''

                # 判断匹配：title、display_text、或中文名部分任一匹配
                combined = f'{title} {display_text}'
                if name in combined or name in title or name in display_text:
                    print(f'  [搜索] "{name}" → celebrity/{celeb_id} (title={title})')
                    return celeb_id

                # 模糊匹配：检查是否第一个非 ASCII 词匹配
                name_parts = name.translate(str.maketrans('·', ' ')).split()
                for part in name_parts:
                    if len(part) >= 2 and (part in combined or part in title):
                        print(f'  [搜索] "{name}" → celebrity/{celeb_id} (partial: {part}, title={title})')
                        return celeb_id

            # 第一条影人结果兜底（可能是英文名导致的匹配失败）
            first_result = soup.select_one('.result a.nbg[href*="/celebrity/"]')
            if first_result:
                href = first_result.get('href', '')
                m = re.search(r'/celebrity/(\d+)/?', href)
                if m:
                    title = first_result.get('title', '')
                    print(f'  [搜索] "{name}" → celebrity/{m.group(1)} (fallback, title={title})')
                    return m.group(1)

            # 直接找所有 celebrity 链接
            for link in soup.select('a[href*="/celebrity/"]'):
                href = link.get('href', '')
                m = re.search(r'/celebrity/(\d+)/?', href)
                text = link.get_text(strip=True)
                if m and text and len(text) >= 2:
                    # 跳过明显的导航链接
                    if text not in ('影人', '更多', '全部', '展开'):
                        print(f'  [搜索] "{name}" → celebrity/{m.group(1)} (link: {text})')
                        return m.group(1)

            return None

        except requests.RequestException as e:
            if attempt < retries - 1:
                print(f'  [搜索] 重试 {attempt+2}/{retries} (超时)...')
                time.sleep(2)
                continue
            print(f'  [搜索] 请求异常 "{name}": {e}')
            return None


def _find_celebrity_from_movies(name: str, movie_ids: list[str]) -> str | None:
    """
    从已知电影页面中提取影人的 celebrity_id
    访问电影页面 → 找到导演/演员链接 → 匹配姓名 → 返回 celebrity_id
    """
    for mid in movie_ids:
        try:
            movie_url = f'https://movie.douban.com/subject/{mid}/'
            resp = requests.get(movie_url, headers=HEADERS, timeout=30, allow_redirects=True)

            if resp.status_code != 200 or len(resp.text) < 5000:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 找所有影人链接: <a href="/celebrity/{id}/">名字</a>
            celeb_links = soup.select('a[href*="/celebrity/"]')
            for link in celeb_links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True)
                m = re.search(r'/celebrity/(\d+)/?', href)
                if m and link_text and link_text == name:
                    print(f'  [兜底] 从电影 {mid} 找到 "{name}" → celebrity/{m.group(1)}')
                    return m.group(1)
                # 部分匹配（处理名字格式差异）
                if m and link_text and (name in link_text or link_text in name):
                    if len(name) >= 2 and len(link_text) >= 2:
                        print(f'  [兜底] 从电影 {mid} 部分匹配 "{name}" ≈ "{link_text}" → celebrity/{m.group(1)}')
                        return m.group(1)

        except requests.RequestException:
            continue

    return None


# ============================================================
# 豆瓣 PoW 验证（SHA-512 工作量证明）
# ============================================================

def solve_pow_challenge(session: requests.Session, html: str) -> requests.Response | None:
    """
    解析豆瓣 PoW 验证页面，计算 SHA-512 哈希，提交验证。
    成功返回跳转后的 Response，失败返回 None。
    """
    import hashlib

    soup = BeautifulSoup(html, 'html.parser')
    tok_el = soup.select_one('#tok')
    cha_el = soup.select_one('#cha')
    red_el = soup.select_one('#red')
    sec_form = soup.select_one('#sec')

    if not tok_el or not cha_el or not red_el:
        return None  # 不是 PoW 验证页

    tok = tok_el.get('value', '')
    cha = cha_el.get('value', '')
    red = red_el.get('value', '')
    action = sec_form.get('action', '/c')

    # 计算 nonce：sha512(cha + nonce) 前 4 位为 0
    difficulty = 4
    target = '0' * difficulty
    nonce = 0
    while True:
        nonce += 1
        h = hashlib.sha512(f'{cha}{nonce}'.encode()).hexdigest()
        if h[:difficulty] == target:
            break
        if nonce % 50000 == 0:
            print(f'  [PoW] 计算中... nonce={nonce}')

    print(f'  [PoW] 完成 nonce={nonce}')

    # 构建提交 URL
    from urllib.parse import urljoin
    submit_url = urljoin('https://movie.douban.com', action)

    resp = session.post(submit_url, data={
        'tok': tok,
        'cha': cha,
        'sol': str(nonce),
        'red': red,
    }, allow_redirects=True, timeout=30)

    return resp


def fetch_page_with_pow(session: requests.Session, url: str, max_retries: int = 3) -> str | None:
    """
    获取页面 HTML，自动处理 PoW 验证链。
    豆瓣的 PoW 验证可能重定向到 sec.douban.com → www.douban.com，需要跟踪完整链条。
    """
    import hashlib

    for attempt in range(max_retries):
        resp = session.get(url, timeout=30, allow_redirects=True)

        if resp.status_code == 200 and len(resp.text) > 5000:
            return resp.text

        if resp.status_code in (200, 403) and len(resp.text) < 5000:
            html = resp.text
            base_url = resp.url  # 当前响应 URL（可能被重定向到 sec.douban.com）

            # 检测 PoW 验证页
            soup = BeautifulSoup(html, 'html.parser')
            tok_el = soup.select_one('#tok')
            cha_el = soup.select_one('#cha')
            red_el = soup.select_one('#red')
            sec_el = soup.select_one('#sec')

            if tok_el and cha_el:
                tok = tok_el.get('value', '')
                cha = cha_el.get('value', '')
                red = red_el.get('value', '') if red_el else ''
                action = sec_el.get('action', '/c') if sec_el else '/c'

                # 从当前 URL 提取 base 用于提交
                from urllib.parse import urlparse, urljoin
                parsed = urlparse(base_url)
                submit_base = f'{parsed.scheme}://{parsed.netloc}'
                submit_url = urljoin(submit_base, action)

                print(f'  [PoW] 验证页 (attempt {attempt+1}), base={submit_base}')

                # 计算 nonce
                difficulty = 4
                target = '0' * difficulty
                nonce = 0
                while True:
                    nonce += 1
                    h = hashlib.sha512(f'{cha}{nonce}'.encode()).hexdigest()
                    if h[:difficulty] == target:
                        break
                    if nonce % 100000 == 0:
                        print(f'    [PoW] 计算中... nonce={nonce}')
                print(f'    [PoW] 完成 nonce={nonce}')

                # 提交验证
                resp2 = session.post(submit_url, data={
                    'tok': tok,
                    'cha': cha,
                    'sol': str(nonce),
                    'red': red,
                }, allow_redirects=True, timeout=30, headers={
                    'Referer': base_url,
                    'Origin': submit_base,
                })

                if resp2.status_code == 200 and len(resp2.text) > 5000:
                    return resp2.text

                # 可能又跳回 PoW 页面（二级验证）
                url = resp2.url  # 用最终的 URL 重试
                continue

    print(f'  [页面] {max_retries} 次尝试后仍无法获取页面')
    return None


# ============================================================
# 解析影人页面
# ============================================================

def parse_celebrity_page(html: str) -> dict:
    """
    解析影人详情页 HTML（www.douban.com/personage/{id}/），提取：
    - avatar_url: 头像链接
    - name: 中文名
    - name_en: 英文名
    - gender: 性别
    - birth_date: 出生日期
    - birthplace: 出生地
    - profession: 职业
    - biography: 个人简介
    """
    soup = BeautifulSoup(html, 'html.parser')
    info = {}

    # 中文名 + 英文名（h1 内格式: "中文名 English Name"）
    h1 = soup.select_one('h1')
    if h1:
        full_name = clean_text(h1.get_text(strip=True))
        # 分离中英文名：找到第一个 ASCII 字符的位置
        split_idx = None
        for i, c in enumerate(full_name):
            if ord(c) < 128 and c.isalpha():
                split_idx = i
                break
        if split_idx is not None and split_idx > 0:
            info['name'] = full_name[:split_idx].strip()
            info['name_en'] = full_name[split_idx:].strip()
        else:
            info['name'] = full_name

    # 头像（personage 页面）
    avatar_img = soup.select_one('img[src*="personage/m/public"]') or \
                 soup.select_one('img[src*="personage/s_ratio"]') or \
                 soup.select_one('.pic img') or \
                 soup.select_one('img[src*="personage"]')
    if avatar_img:
        src = avatar_img.get('src', '')
        if src:
            # 替换为高清大图: s→l, m→l
            src = re.sub(r'/personage/[sm]/', '/personage/l/', src)
            src = src.replace('/personage/small/', '/personage/l/')
            info['avatar_url'] = src

    # 属性列表（性别、出生日期、出生地、职业等）
    for li in soup.select('.subject-property li') or soup.select('.info-list li') or soup.select('.basic-info li'):
        label_el = li.select_one('.label')
        value_el = li.select_one('.value')
        if not label_el or not value_el:
            continue
        label = clean_text(label_el.get_text(strip=True)).rstrip(':：')
        val = clean_text(value_el.get_text(strip=True))
        if not label or not val:
            continue

        if '性别' in label:
            info['gender'] = val
        elif '出生日期' in label:
            info['birth_date'] = val
        elif '出生地' in label or '生地' in label:
            info['birthplace'] = val
        elif '职业' in label:
            info['profession'] = val
        elif '英文名' in label or '外文名' in label:
            info['name_en'] = val

    # 个人简介（在 .desc 或 #intro 中）
    desc_el = soup.select_one('.desc') or \
              soup.select_one('#intro .bd') or \
              soup.select_one('.intro .bd')
    if desc_el:
        # 去掉展开/收起按钮
        for tag in desc_el.select('a, button, .toggle-more, .expand'):
            tag.decompose()
        bio = clean_text(desc_el.get_text(' ', strip=True))
        # 去掉开头的名字（通常简介以名字开头）
        if info.get('name') and bio.startswith(info['name']):
            # 保留简介内容
            pass
        if len(bio) > 50:
            info['biography'] = bio[:2000]

    return info


# ============================================================
# 下载头像
# ============================================================

def download_avatar(avatar_url: str, celeb_id: str) -> str | None:
    """
    下载高清头像到 static/persons/{celeb_id}.jpg
    返回本地相对路径，失败返回 None
    """
    if not avatar_url:
        return None

    ensure_dir(AVATAR_DIR)

    # 确定扩展名（豆瓣影人头像基本是 jpg）
    ext = '.jpg'
    if '.png' in avatar_url:
        ext = '.png'
    elif '.webp' in avatar_url and '.jpg' not in avatar_url and '.jpeg' not in avatar_url:
        ext = '.webp'

    local_path = AVATAR_DIR / f'{celeb_id}{ext}'

    # 已存在的大文件（>5KB 才是有效头像）跳过
    if local_path.exists() and local_path.stat().st_size > 5120:
        print(f'  [头像] 已存在，跳过 → {local_path.name}')
        return f'persons/{celeb_id}{ext}'

    # 如果存在但文件过小（上次下载失败/无效），先删掉重下
    if local_path.exists():
        local_path.unlink()

    try:
        resp = requests.get(avatar_url, headers=IMAGE_HEADERS, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 5120:
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            print(f'  [头像] 下载成功 → {local_path.name} ({len(resp.content)} bytes)')
            return f'persons/{celeb_id}{ext}'
        elif resp.status_code == 404:
            print(f'  [头像] 404 无头像')
            return None
        else:
            print(f'  [头像] HTTP {resp.status_code} / 文件过小({len(resp.content)}b)，跳过')
            return None
    except requests.RequestException as e:
        print(f'  [头像] 下载异常: {e}')
        return None
        return None


# ============================================================
# 主入口：爬取单个影人
# ============================================================

def scrape_person(name: str, cache: dict | None = None, known_movies: list[str] | None = None) -> dict | None:
    """
    爬取一个影人的完整信息
    - name: 影人姓名
    - cache: 已有缓存字典（用于增量更新）
    - known_movies: 该影人参演的电影 ID 列表（搜索失败时从电影页面兜底）
    返回影人信息字典，失败返回 None
    """
    if cache is None:
        cache = load_persons_cache()

    # 命中缓存（有头像路径或明确标记失败）
    if name in cache:
        cached = cache[name]
        if cached and cached.get('avatar_local_path'):
            print(f'  [缓存] "{name}" 已存在，跳过')
            return cached
        # 之前标记为失败（scrape_failed=True）
        if cached and cached.get('_scrape_failed'):
            print(f'  [缓存] "{name}" 上次爬取失败，跳过')
            return None

    # 1. 搜索影人 ID（带电影兜底）
    celeb_id = search_celebrity(name, known_movies=known_movies)
    if not celeb_id:
        # 标记失败，避免重复搜索
        cache[name] = {'_scrape_failed': True}
        save_persons_cache(cache)
        return None

    random_delay()

    # 2. 获取影人页面
    # movie.douban.com/celebrity/{id} → 302 → www.douban.com/personage/{pid}
    session = requests.Session()
    session.headers.update(HEADERS)

    personage_id = None
    html = None

    # 第一步：访问 movie.douban.com/celebrity/{id}，不要跟随重定向
    celeb_url = f'https://movie.douban.com/celebrity/{celeb_id}/'
    try:
        resp = session.get(celeb_url, allow_redirects=False, timeout=30)
        if resp.status_code in (301, 302):
            location = resp.headers.get('Location', '')
            m = re.search(r'/personage/(\d+)', location)
            if m:
                personage_id = m.group(1)
                print(f'  [重定向] celebrity/{celeb_id} → personage/{personage_id}')
    except requests.RequestException as e:
        print(f'  [页面] 请求异常: {e}')
        return None

    if personage_id:
        # 第二步：直接访问 www.douban.com/personage/{id}
        personage_url = f'https://www.douban.com/personage/{personage_id}/'
        try:
            resp = session.get(personage_url, allow_redirects=True, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 5000:
                html = resp.text
            elif len(resp.text) < 5000 and ('sha512' in resp.text or 'tok' in resp.text):
                # PoW 验证
                html = fetch_page_with_pow(session, personage_url)
        except requests.RequestException as e:
            print(f'  [页面] 请求异常: {e}')

    if not html:
        print(f'  [页面] 无法获取 "{name}" 的影人页面')
        return None

    # 3. 解析页面
    info = parse_celebrity_page(html)
    info['douban_id'] = celeb_id
    info['personage_id'] = personage_id
    info['_scraped_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # 如果页面解析出的名字为空，用搜索名补上
    if not info.get('name'):
        info['name'] = name

    print(f'  [解析] {info.get("name", name)} | {info.get("gender","")} | {info.get("birth_date","")} | {info.get("profession","")} | bio={bool(info.get("biography"))} | avatar={bool(info.get("avatar_url"))}')

    random_delay()

    # 4. 下载头像
    avatar_url = info.get('avatar_url', '')
    if avatar_url:
        local_path = download_avatar(avatar_url, personage_id or celeb_id)
        if local_path:
            info['avatar_local_path'] = local_path

    # 5. 存入缓存
    cache[name] = info
    save_persons_cache(cache)

    return info


# ============================================================
# 批量爬取
# ============================================================

def extract_all_persons() -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """
    从电影数据中提取所有导演和演员
    返回 (directors, actors)，每项为 (name, [movie_ids])，按出现频率降序排列
    """
    with open(MOVIE_DATA_FILE, 'r', encoding='utf-8') as f:
        movies = json.load(f)

    director_data = {}   # name → {count, movie_ids}
    actor_data = {}

    for m in movies:
        mid = str(m.get('id', ''))
        title = m.get('title', '')
        if not mid or not title:
            continue

        if m.get('director'):
            for d in m['director'].replace('/', ',').split(','):
                d = d.strip()
                if d:
                    if d not in director_data:
                        director_data[d] = {'count': 0, 'mids': []}
                    director_data[d]['count'] += 1
                    director_data[d]['mids'].append(mid)

        if m.get('cast'):
            for a in m['cast'].replace('/', ',').split(','):
                a = a.strip()
                if a:
                    if a not in actor_data:
                        actor_data[a] = {'count': 0, 'mids': []}
                    actor_data[a]['count'] += 1
                    # 限制每个人的电影数量，避免数据膨胀
                    if len(actor_data[a]['mids']) < 10:
                        actor_data[a]['mids'].append(mid)

    # 按出现次数降序排列
    directors = sorted(director_data.items(), key=lambda x: x[1]['count'], reverse=True)
    actors = sorted(actor_data.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f'导演: {len(directors)} 人, 演员: {len(actors)} 人')
    print(f"Top 10 导演: {[(d[0], d[1]['count']) for d in directors[:10]]}")
    print(f"Top 10 演员: {[(a[0], a[1]['count']) for a in actors[:10]]}")

    return ([(d[0], d[1]['mids']) for d in directors],
            [(a[0], a[1]['mids']) for a in actors])


def batch_scrape(directors: list[tuple[str, list[str]]] | None = None,
                 actors: list[tuple[str, list[str]]] | None = None,
                 max_actors: int = 3000,
                 start_index: int = 0):
    """
    批量爬取影人数据
    - directors: 导演名单 [(name, [movie_ids]), ...]
    - actors: 演员名单 [(name, [movie_ids]), ...]
    - max_actors: 最多爬取多少个演员（按频率取 top N）
    - start_index: 从哪个索引开始（断点续传）
    """
    cache = load_persons_cache()
    print(f'已缓存: {len(cache)} 人')

    # 构建待爬列表：导演全部爬，演员爬 top N
    todo = []
    if directors:
        todo.extend([(name, 'director', mids) for name, mids in directors])
    if actors:
        todo.extend([(name, 'actor', mids) for name, mids in actors[:max_actors]])

    total = len(todo)
    print(f'计划爬取: {total} 人 (导演 {len(directors) if directors else 0} + 演员 {min(len(actors) if actors else 0, max_actors)})')
    print(f'起始索引: {start_index}')

    success = 0
    failed = 0
    skipped = 0

    try:
        for i in range(start_index, total):
            name, ptype, mids = todo[i]
            idx = i + 1
            print(f'\n[{idx}/{total}] [{ptype}] {name}')

            # 检查缓存
            if name in cache:
                cached = cache[name]
                if cached.get('avatar_local_path'):
                    skipped += 1
                    continue
                if cached.get('_scrape_failed'):
                    skipped += 1
                    continue

            result = scrape_person(name, cache, known_movies=mids)
            if result and result.get('avatar_local_path'):
                success += 1
            elif result and not result.get('avatar_local_path'):
                # 有信息但没头像（也算部分成功）
                success += 1
            else:
                failed += 1

            # 每 50 条打印进度 & 强制保存
            if idx % 50 == 0:
                save_persons_cache(cache)
                print(f'\n--- 进度: {idx}/{total} | 成功: {success} | 失败: {failed} | 跳过: {skipped} | [已保存] ---')

            random_delay()

    except KeyboardInterrupt:
        print(f'\n用户中断！当前索引进度: {i}')
        print(f'下次可从 start_index={i} 继续')
        print(f'缓存已保存到 {PERSONS_DATA_FILE}')

    finally:
        save_persons_cache(cache)
        print(f'\n=== 批量爬取结束 ===')
        print(f'总计: {total} | 成功: {success} | 失败: {failed} | 缓存跳过: {skipped}')
        print(f'缓存文件: {PERSONS_DATA_FILE}')
        print(f'头像目录: {AVATAR_DIR}')


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import sys

    # 确保目录存在
    ensure_dir(AVATAR_DIR)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 测试模式：爬取几个人看看效果
        test_names = ['张艺谋', '肖麓西', '马思纯', '白客']
        print('=== 测试模式 ===')
        for name in test_names:
            print(f'\n>>> 爬取: {name}')
            info = scrape_person(name)
            if info:
                print(f'    结果: {json.dumps(info, ensure_ascii=False, indent=2)}')
            random_delay()

    elif len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # 批量模式
        directors, actors = extract_all_persons()
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        batch_scrape(directors=directors, actors=actors, max_actors=3000, start_index=start)

    elif len(sys.argv) > 1 and sys.argv[1] == 'directors':
        # 只爬导演
        directors, _ = extract_all_persons()
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        batch_scrape(directors=directors, actors=[], start_index=start)

    elif len(sys.argv) > 1 and sys.argv[1] == 'actors':
        # 只爬演员
        _, actors = extract_all_persons()
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        batch_scrape(directors=[], actors=actors, max_actors=3000, start_index=start)

    elif len(sys.argv) > 1 and sys.argv[1] == 'stats':
        # 查看缓存统计
        cache = load_persons_cache()
        total = len(cache)
        with_avatar = sum(1 for v in cache.values() if v.get('avatar_local_path'))
        with_bio = sum(1 for v in cache.values() if v.get('biography'))
        failed = sum(1 for v in cache.values() if v.get('_scrape_failed'))
        print(f'缓存统计: 总计 {total} | 有头像 {with_avatar} | 有简介 {with_bio} | 失败标记 {failed}')

    else:
        print('豆瓣影人爬虫 - 用法:')
        print('  python douban_scraper.py test       # 测试模式，爬取几个示例')
        print('  python douban_scraper.py directors   # 爬取所有导演')
        print('  python douban_scraper.py actors      # 爬取 Top 3000 演员')
        print('  python douban_scraper.py batch [N]   # 爬取导演+Top3000演员，从第N个开始')
        print('  python douban_scraper.py stats       # 查看缓存统计')
