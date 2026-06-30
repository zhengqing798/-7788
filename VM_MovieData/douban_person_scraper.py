#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
豆瓣影人数据爬虫
- 只爬取出现次数 ≥ 2 的导演和演员
- 请求间隔 2~5 秒（与电影爬虫一致，避免风控）
- 搜索影人 → 302重定向 → personage页面 → 解析详情 → 下载头像
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
from urllib.parse import quote

# Windows 终端 UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ====================== 配置 ======================
DELAY_RANGE = (2, 5)                # 请求间隔，与电影爬虫一致
TIMEOUT = 20                        # 请求超时
MAX_RETRIES = 3                     # 最大重试
MIN_APPEARANCES = 2                 # 最少出现次数

BASE_DIR = Path(__file__).parent
MOVIE_DATA = BASE_DIR / 'movies_data_temp(1).json'
PERSONS_FILE = BASE_DIR / 'persons_data.json'
AVATAR_DIR = BASE_DIR / 'static' / 'persons'

# 豆瓣 Cookie（从浏览器完整复制）
DOUBAN_COOKIES = {
    'bid': 'SlJWO_1i1zs',
    'dbcl2': '"295727633:9giEOdx27zw"',
    'ck': '-aKW',
    'll': '"118201"',
    '__utma': '30149280.965526527.1782112451.1782368946.1782377102.8',
    '__utmb': '30149280.0.10.1782377102',
    '__utmc': '30149280',
    '__utmv': '30149280.29572',
    '__utmz': '30149280.1782377102.8.7.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/',
    '_pk_id.100001.8cb4': '8220a74c02256fb2.1782088954.',
    '_pk_ref.100001.8cb4': '%5B%22%22%2C%22%22%2C1782371963%2C%22https%3A%2F%2Fcn.bing.com%2F%22%5D',
    '_vwo_uuid_v2': 'DB701F7847911E41D24D86CC09E875651|ddcc09ff5e061aad38521f1e7d9639b7',
    'ap_v': '0,6.0',
    'frodotk_db': '"bae904c701e3493230bb0e5505a9db96"',
    'push_doumail_num': '0',
    'push_noty_num': '0',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
}

IMAGE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://movie.douban.com/',
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

os.makedirs(AVATAR_DIR, exist_ok=True)


# ====================== 工具函数 ======================
def safe_delay(msg=None):
    delay = random.uniform(*DELAY_RANGE)
    if msg:
        print(f'  [等待] {delay:.1f}s {msg}')
    time.sleep(delay)


def create_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update(DOUBAN_COOKIES)
    return s


def load_cache():
    if PERSONS_FILE.exists():
        with open(PERSONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(data):
    with open(PERSONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ====================== 提取影人列表 ======================
def build_person_list():
    """提取出现次数 ≥ MIN_APPEARANCES 的导演和演员"""
    with open(MOVIE_DATA, 'r', encoding='utf-8') as f:
        movies = json.load(f)

    person_freq = {}
    person_movies = {}   # name → [movie_ids]

    for m in movies:
        mid = str(m.get('id', ''))
        if not mid or not m.get('title'):
            continue

        seen = set()
        for field in ['director', 'cast']:
            val = m.get(field, '')
            if not val:
                continue
            for name in val.replace('/', ',').split(','):
                name = name.strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                person_freq[name] = person_freq.get(name, 0) + 1
                if name not in person_movies:
                    person_movies[name] = []
                if len(person_movies[name]) < 10:
                    person_movies[name].append(mid)

    # 筛选 ≥ 2 次
    qualified = [(name, cnt, person_movies.get(name, []))
                 for name, cnt in person_freq.items() if cnt >= MIN_APPEARANCES]
    qualified.sort(key=lambda x: x[1], reverse=True)

    print(f'总影人: {len(person_freq)}')
    print(f'出现≥{MIN_APPEARANCES}次: {len(qualified)}')
    print(f'Top 20: {[(n, c) for n, c, _ in qualified[:20]]}')
    return qualified


# ====================== 搜索影人 ======================
def search_celebrity(session, name):
    """搜索影人ID"""
    url = f'https://movie.douban.com/celebrities/search?search_text={quote(name)}'

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(3)
                    continue
                return None

            soup = BeautifulSoup(resp.text, 'html.parser')

            for result in soup.select('.result'):
                nbg = result.select_one('a.nbg[href*="/celebrity/"]')
                if not nbg:
                    continue
                href = nbg.get('href', '')
                m = re.search(r'/celebrity/(\d+)/?', href)
                if not m:
                    continue
                celeb_id = m.group(1)
                title = nbg.get('title', '')
                display = (result.select_one('.content h3 a') or result.select_one('h3 a'))
                display_text = display.get_text(strip=True) if display else ''

                combined = f'{title} {display_text}'
                if name in combined or name in title:
                    print(f'  [搜索] "{name}" → celebrity/{celeb_id}')
                    return celeb_id

                # 分词匹配
                for part in name.replace('·', ' ').split():
                    if len(part) >= 2 and part in combined:
                        print(f'  [搜索] "{name}" → celebrity/{celeb_id} (part={part})')
                        return celeb_id

            # 第一条兜底
            first = soup.select_one('a.nbg[href*="/celebrity/"]')
            if first:
                href = first.get('href', '')
                m = re.search(r'/celebrity/(\d+)/?', href)
                if m:
                    print(f'  [搜索] "{name}" → celebrity/{m.group(1)} (fallback)')
                    return m.group(1)

            return None

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f'  [搜索] 重试 {attempt+2}/{MAX_RETRIES}: {e}')
                time.sleep(3)
                continue
            return None


# ====================== 从电影页面兜底搜索 ======================
def find_celebrity_from_movie(session, name, movie_ids):
    """搜索失败时，从已知电影页面提取影人链接"""
    for mid in movie_ids[:5]:
        try:
            url = f'https://movie.douban.com/subject/{mid}/'
            resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code != 200 or len(resp.text) < 5000:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.select('a[href*="/celebrity/"]'):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                m = re.search(r'/celebrity/(\d+)/?', href)
                if m and text and (text == name or name in text or text in name):
                    print(f'  [兜底] 从电影{mid}找到 "{name}" → celebrity/{m.group(1)}')
                    return m.group(1)

        except requests.RequestException:
            continue
    return None


# ====================== 获取影人详情页 ======================
def fetch_personage_page(session, celeb_id):
    """
    访问 movie.douban.com/celebrity/{id}
    → 302 → www.douban.com/personage/{pid}
    → 返回 HTML
    """
    # Step 1: 不跟随重定向，获取 personage_id
    try:
        resp = session.get(
            f'https://movie.douban.com/celebrity/{celeb_id}/',
            allow_redirects=False, timeout=TIMEOUT
        )
        personage_id = None
        if resp.status_code in (301, 302):
            loc = resp.headers.get('Location', '')
            m = re.search(r'/personage/(\d+)', loc)
            if m:
                personage_id = m.group(1)
    except requests.RequestException:
        return None, None

    if not personage_id:
        return None, None

    # Step 2: 访问 personage 页面
    try:
        resp = session.get(
            f'https://www.douban.com/personage/{personage_id}/',
            allow_redirects=True, timeout=TIMEOUT
        )
        if resp.status_code == 200 and len(resp.text) > 5000:
            return resp.text, personage_id
    except requests.RequestException:
        pass

    return None, personage_id


# ====================== 解析影人信息 ======================
def parse_personage(html):
    """解析 personage 页面"""
    soup = BeautifulSoup(html, 'html.parser')
    info = {}

    # 名字
    h1 = soup.select_one('h1')
    if h1:
        full = h1.get_text(strip=True)
        # 分离中文名和英文名
        split_at = None
        for i, c in enumerate(full):
            if ord(c) < 128 and c.isalpha():
                split_at = i
                break
        if split_at and split_at > 0:
            info['name'] = full[:split_at].strip()
            info['name_en'] = full[split_at:].strip()
        else:
            info['name'] = full

    # 头像 URL — 优先 .avatar-container 中的 img.avatar
    avatar = (soup.select_one('.avatar-container img.avatar') or
              soup.select_one('img.avatar') or
              soup.select_one('img[src*="personage/m/public"]') or
              soup.select_one('img[src*="personage/s_ratio"]') or
              soup.select_one('.pic img') or
              soup.select_one('img[src*="celebrity/m/public"]') or
              soup.select_one('img[src*="personage"]'))
    if avatar:
        src = avatar.get('src', '')
        if src:
            # 统一转为高清大图: /m/ → /l/ 或 /raw/
            # celebrity 路径: /celebrity/m/public/ → /celebrity/raw/public/
            # personage 路径: /personage/m/public/ → /personage/l/public/
            if '/celebrity/' in src:
                src = re.sub(r'/celebrity/[ms]/', '/celebrity/raw/', src)
            else:
                src = re.sub(r'/personage/[sm]/', '/personage/l/', src)
            info['avatar_url'] = src

    # 属性列表
    for li in soup.select('.subject-property li'):
        label_el = li.select_one('.label')
        value_el = li.select_one('.value')
        if not label_el or not value_el:
            continue
        label = label_el.get_text(strip=True).rstrip(':：')
        val = value_el.get_text(strip=True)
        if not val:
            continue
        if '性别' in label:
            info['gender'] = val
        elif '出生日期' in label:
            info['birth_date'] = val
        elif '出生地' in label:
            info['birthplace'] = val
        elif '职业' in label:
            info['profession'] = val
        elif '英文名' in label or '外文名' in label:
            info['name_en'] = val

    # 简介
    desc = soup.select_one('.desc') or soup.select_one('#intro .bd')
    if desc:
        for tag in desc.select('a, button, .toggle-more'):
            tag.decompose()
        bio = desc.get_text(' ', strip=True)
        if len(bio) > 50:
            info['biography'] = bio[:2000]

    return info


# ====================== 下载头像 ======================
def download_avatar(avatar_url, person_id):
    """下载头像，自动处理 celebrity 和 personage 两种路径"""
    if not avatar_url or 'img9.doubanio.com' in avatar_url:
        return None

    ext = '.webp' if '.webp' in avatar_url else '.jpg'
    path = AVATAR_DIR / f'{person_id}{ext}'

    if path.exists() and path.stat().st_size > 3072:
        return f'persons/{person_id}{ext}'
    if path.exists():
        path.unlink()

    # celebrity 路径: /raw/ → /m/ → /s/
    # personage 路径: /l/ → /m/
    if '/celebrity/' in avatar_url:
        size_candidates = ['raw', 'm', 's']
        size_labels = {'raw': '原图', 'm': '中图', 's': '小图'}
    else:
        size_candidates = ['l', 'm']
        size_labels = {'l': '高清', 'm': '中图'}

    for size in size_candidates:
        url = re.sub(r'/(celebrity|personage)/[a-z]+/', f'/\\1/{size}/', avatar_url)
        try:
            resp = requests.get(url, headers=IMAGE_HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 3072:
                with open(path, 'wb') as f:
                    f.write(resp.content)
                print(f'  [头像] {size_labels.get(size, size)} {path.name} ({len(resp.content)} bytes)')
                return f'persons/{person_id}{ext}'
        except requests.RequestException:
            continue
    return None


# ====================== 爬取单人 ======================
def scrape_person(name, movie_ids, cache):
    """爬取单个影人"""
    if name in cache and cache[name].get('douban_id'):
        cached = cache[name]
        if cached.get('avatar_local_path'):
            return cached
        if cached.get('_no_avatar'):
            return cached

    session = create_session()

    # 1. 搜索
    safe_delay()
    celeb_id = search_celebrity(session, name)
    if not celeb_id:
        celeb_id = find_celebrity_from_movie(session, name, movie_ids)
    if not celeb_id:
        print(f'  [跳过] 未找到')
        return None

    # 2. 获取详情页
    safe_delay()
    html, personage_id = fetch_personage_page(session, celeb_id)
    if not html:
        print(f'  [跳过] 无法获取页面')
        return None

    # 3. 解析
    info = parse_personage(html)
    info['douban_id'] = celeb_id
    info['personage_id'] = personage_id
    info['_scraped_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    if not info.get('name'):
        info['name'] = name

    has_avatar = bool(info.get('avatar_url'))
    has_bio = bool(info.get('biography'))
    print(f'  [解析] {info["name"]} | {info.get("gender","")} | {info.get("birth_date","")} | '
          f'{info.get("profession","")} | avatar={has_avatar} | bio={has_bio}')

    # 4. 下载头像
    if info.get('avatar_url'):
        safe_delay()
        local = download_avatar(info['avatar_url'], personage_id or celeb_id)
        if local:
            info['avatar_local_path'] = local

    if not info.get('avatar_local_path'):
        info['_no_avatar'] = True

    # 5. 保存
    cache[name] = info
    save_cache(cache)
    return info


# ====================== 批量爬取 ======================
def batch_scrape(start_index=0):
    person_list = build_person_list()
    cache = load_cache()
    total = len(person_list)

    print('=' * 60)
    print(f'豆瓣影人数据爬虫')
    print(f'目标数量: {total} 人（出现≥{MIN_APPEARANCES}次）')
    print(f'已缓存: {len(cache)} 人')
    print(f'起始索引: {start_index}')
    print(f'请求间隔: {DELAY_RANGE[0]}~{DELAY_RANGE[1]}s')
    est_hours = (total - start_index) * 4.5 * sum(DELAY_RANGE) / 2 / 3600
    print(f'预计剩余: {est_hours:.1f} 小时')
    print('=' * 60)

    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    try:
        for i in range(start_index, total):
            name, freq, movie_ids = person_list[i]
            idx = i + 1
            pct = idx / total * 100
            elapsed = time.time() - start_time
            rate = (idx - start_index) / elapsed * 60 if elapsed > 0 else 0
            eta = (total - idx) / rate * 60 if rate > 0 else 0

            print(f'\n处理第 {idx}/{total} 人 ({pct:.1f}%) [{freq}部]: {name}'
                  f'  [速率: {rate:.1f}人/分 | 预计剩余: {eta/60:.1f}h]')

            if name in cache:
                c = cache[name]
                if c.get('avatar_local_path') or c.get('_no_avatar'):
                    skipped += 1
                    continue

            try:
                result = scrape_person(name, movie_ids, cache)
                if result and result.get('douban_id'):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f'  [异常] {e}')
                failed += 1
                safe_delay()

            # 每50条保存一次
            if idx % 50 == 0:
                save_cache(cache)
                avatars = sum(1 for v in cache.values() if v.get('avatar_local_path'))
                print(f'\n--- 已保存中间结果: 总计 {len(cache)} 条 | 成功 {success} | '
                      f'失败 {failed} | 跳过 {skipped} | 头像 {avatars} ---')

    except KeyboardInterrupt:
        save_cache(cache)
        print(f'\n中断于第 {i} 人，数据已保存')
    finally:
        save_cache(cache)
        total_time = (time.time() - start_time) / 3600
        avatars = sum(1 for v in cache.values() if v.get('avatar_local_path'))
        print(f'\n{"="*60}')
        print(f'爬取结束: 总计 {len(cache)} | 成功 {success} | 失败 {failed} | 跳过 {skipped}')
        print(f'有头像: {avatars} | 耗时: {total_time:.1f}h')
        print(f'缓存文件: {PERSONS_FILE}')
        print(f'头像目录: {AVATAR_DIR}')
        print(f'{"="*60}')


# ====================== 命令行入口 ======================
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print('=== 测试模式：爬取 10 人 ===')
        person_list = build_person_list()
        cache = load_cache()
        for name, freq, movie_ids in person_list[:10]:
            print(f'\n>>> [{freq}部] {name}')
            result = scrape_person(name, movie_ids, cache)
            if result:
                keep = {k: v for k, v in result.items() if not k.startswith('_')}
                print(f'  结果: {json.dumps(keep, ensure_ascii=False, indent=2)}')
    elif len(sys.argv) > 1 and sys.argv[1] == 'batch':
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        batch_scrape(start_index=start)
    elif len(sys.argv) > 1 and sys.argv[1] == 'stats':
        cache = load_cache()
        total = len(cache)
        avatars = sum(1 for v in cache.values() if v.get('avatar_local_path'))
        bios = sum(1 for v in cache.values() if v.get('biography'))
        print(f'缓存: 总计 {total} | 有头像 {avatars} | 有简介 {bios}')
    else:
        print('豆瓣影人爬虫')
        print('  python douban_person_scraper.py test    # 测试 10 人')
        print('  python douban_person_scraper.py batch   # 批量爬取')
        print('  python douban_person_scraper.py stats   # 查看统计')
