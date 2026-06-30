#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import random
import re
import os
import json
import csv
from pyquery import PyQuery as pq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====================== 配置 ======================
TARGET_COUNT = 10000               # 目标数量
MAX_RETRIES = 5                    # 最大重试次数
DELAY_RANGE = (2, 5)               # 请求间隔（秒）

# ---------- 模拟真实浏览器请求头 ----------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Connection': 'keep-alive',
    'Referer': 'https://movie.douban.com/',
    'Origin': 'https://movie.douban.com',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}

# ---------- 直接从您提供的 Cookie 中提取关键字段 ----------
COOKIES = {
    'bid': 'SlJWO_1i1zs',
    'dbcl2': '"295727633:9giEOdx27zw"',   # 注意原值带双引号
    'ck': '-aKW',
    'll': '"118201"',                     # 注意原值带双引号
    '__utma': '30149280.965526527.1782112451.1782112451.1782112451.1',
    '__utmb': '30149280.2.10.1782112451',
    '__utmc': '30149280',
    '__utmt': '1',
    '__utmv': '30149280.29572',
    '__utmz': '30149280.1782112451.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
    '__yadk_uid': '9SOYkbEx7FFT60puiV0ZWVeVtwnLcFlV',
    '_pk_id.100001.4cf6': 'ef9ed9faf23cb928.1782094300.',
    '_pk_ref.100001.4cf6': '%5B%22%22%2C%22%22%2C1782112454%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D',
    '_pk_ses.100001.4cf6': '1',
    '_vwo_uuid_v2': 'DB701F7847911E41D24D86CC09E875651|ddcc09ff5e061aad38521f1e7d9639b7',
    'ap_v': '0,6.0',
    'frodotk_db': '"7d36ec9767a92a0677a4c5c636fb7e12"',
    'push_doumail_num': '0',
    'push_noty_num': '0',
}

# ---------- 代理（可选） ----------
PROXY_LIST = [
    # 'http://127.0.0.1:8080',
]

POSTER_DIR = 'posters'
os.makedirs(POSTER_DIR, exist_ok=True)

# ====================== Session 工厂 ======================
def create_session(proxy=None):
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(COOKIES)
    if proxy:
        session.proxies = {'http': proxy, 'https': proxy}
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# ====================== 预热 ======================
def warmup(proxy=None):
    session = create_session(proxy)
    try:
        print("🌐 预热：访问豆瓣电影首页...")
        resp = session.get('https://movie.douban.com/', timeout=10)
        if resp.status_code == 200:
            if 'error code: 004' in resp.text or 'please Login' in resp.text:
                print("❌ 预热失败：Cookie 无效，请检查是否过期。")
                return None
            print("✅ 预热成功，会话已建立。")
            return session
        else:
            print(f"⚠️ 预热失败，状态码: {resp.status_code}")
            return None
    except Exception as e:
        print(f"❌ 预热异常: {e}")
        return None

# ====================== 获取标签 ======================
def get_tags(proxy=None):
    sess = warmup(proxy)
    if not sess:
        sess = create_session(proxy)
    url = 'https://movie.douban.com/j/search_tags?type=movie'
    for attempt in range(3):
        try:
            time.sleep(random.uniform(1, 3))
            resp = sess.get(url, timeout=10)
            if resp.status_code == 403 or ('error code' in resp.text):
                print(f"⚠️ 第{attempt+1}次获取标签失败（可能Cookie失效）。")
                if attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"⏳ 等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    # 重新创建 session（可能 Cookie 已更新）
                    sess = create_session(proxy)
                    continue
                else:
                    return []
            if resp.status_code != 200:
                print(f"获取标签失败，状态码: {resp.status_code}")
                continue
            data = resp.json()
            tags = data.get('tags', [])
            print(f"✅ 获取到标签: {tags}")
            return tags
        except Exception as e:
            print(f"获取标签异常: {e}")
            if attempt < 2:
                time.sleep(20 * (attempt + 1))
    return []

# ====================== 获取电影列表（分页） ======================
def get_movies_by_tag(tag, proxy=None, page_limit=100):
    session = create_session(proxy)
    movies = []
    page_start = 0
    while True:
        url = f'https://movie.douban.com/j/search_subjects?type=movie&tag={tag}&page_limit={page_limit}&page_start={page_start}'
        try:
            time.sleep(random.uniform(0.5, 1.5))
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"获取 {tag} 分页 {page_start} 失败，状态码: {resp.status_code}")
                break
            data = resp.json()
            subjects = data.get('subjects', [])
            if not subjects:
                break
            for item in subjects:
                movies.append({
                    'id': item['id'],
                    'title': item['title'],
                    'cover': item['cover'],
                    'rate': item.get('rate', ''),
                })
            if len(subjects) < page_limit:
                break
            page_start += page_limit
        except Exception as e:
            print(f"获取标签 {tag} 分页 {page_start} 出错: {e}")
            break
    return movies

def collect_movie_ids(target, proxy=None):
    all_movies = []
    movie_ids = set()
    tags = get_tags(proxy)
    if not tags:
        print("⚠️ 未能获取到标签，请检查 Cookie 是否有效。")
        return []
    for tag in tags:
        if len(all_movies) >= target:
            break
        print(f"正在处理标签: {tag}")
        movies = get_movies_by_tag(tag, proxy)
        for m in movies:
            if m['id'] not in movie_ids:
                movie_ids.add(m['id'])
                all_movies.append(m)
                if len(all_movies) >= target:
                    break
        print(f"已收集 {len(all_movies)} 部电影")
        time.sleep(random.uniform(1, 3))
    return all_movies

# ====================== 解析详情页 ======================
def parse_detail(movie_id, movie_info, proxy=None):
    session = create_session(proxy)
    url = f'https://movie.douban.com/subject/{movie_id}/'
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            resp = session.get(url, timeout=20)
            resp.encoding = 'utf-8'
            resp.raise_for_status()
            if 'sec.douban.com' in resp.url:
                print(f"被重定向到验证页面，{movie_id} 跳过")
                return None
            doc = pq(resp.text)

            # 从 #info 文本提取
            info_text = doc('#info').text()
            info_dict = {}
            if info_text:
                pattern = re.compile(r'([^:]+):\s*([^\n]+)')
                for match in pattern.findall(info_text):
                    key = match[0].strip()
                    value = match[1].strip()
                    if key and value and key not in ['更多']:
                        info_dict[key] = value

            # 后备选择器
            if not info_dict:
                director_elem = doc('a[rel="v:directedBy"]')
                if director_elem:
                    info_dict['导演'] = director_elem.text()
                starring_elems = doc('a[rel="v:starring"]')
                if starring_elems:
                    info_dict['主演'] = ' / '.join([a.text() for a in starring_elems.items()])
                genre_elems = doc('span[property="v:genre"]')
                if genre_elems:
                    info_dict['类型'] = ' / '.join([g.text() for g in genre_elems.items()])

            summary = doc('span[property="v:summary"]').text().strip()
            if not summary:
                summary = doc('#link-report .all-hidden').text().strip()
            if not summary:
                summary = doc('#link-report .related-info').text().strip()

            rating = movie_info.get('rate', '')
            if not rating:
                rating = doc('strong[property="v:average"]').text().strip()

            cover_url = movie_info.get('cover', '')
            if not cover_url:
                cover_url = doc('img[rel="v:image"]').attr('src')

            detail = {
                'id': movie_id,
                'title': movie_info['title'],
                'cover': cover_url,
                'rating': rating,
                'director': info_dict.get('导演', ''),
                'screenwriter': info_dict.get('编剧', ''),
                'cast': info_dict.get('主演', ''),
                'genre': info_dict.get('类型', ''),
                'country': info_dict.get('制片国家/地区', ''),
                'language': info_dict.get('语言', ''),
                'release_date': info_dict.get('上映日期', ''),
                'runtime': info_dict.get('片长', ''),
                'imdb': info_dict.get('IMDb', ''),
                'aka': info_dict.get('又名', ''),
                'summary': summary,
            }
            non_empty = {k: v for k, v in detail.items() if v}
            print(f"✅ 提取到 {len(non_empty)} 个字段")
            return detail
        except Exception as e:
            print(f"解析 {movie_id} 第{attempt+1}次失败: {e}")
            wait = 5 * (attempt + 1)
            print(f"⏳ 等待 {wait} 秒后重试...")
            time.sleep(wait)
    print(f"解析 {movie_id} 多次失败，放弃")
    return None

# ====================== 短评和海报 ======================
def get_hot_comments(movie_id, proxy=None):
    session = create_session(proxy)
    url = f'https://movie.douban.com/subject/{movie_id}/comments'
    try:
        time.sleep(random.uniform(0.3, 0.8))
        resp = session.get(url, timeout=15)
        resp.encoding = 'utf-8'
        resp.raise_for_status()
        if 'sec.douban.com' in resp.url:
            return []
        doc = pq(resp.text)
        comments = []
        for item in doc('.comment-item').items():
            if len(comments) >= 5:
                break
            content = item('.short').text().strip()
            if content:
                comments.append(content)
        return comments
    except Exception as e:
        print(f"获取短评 {movie_id} 出错: {e}")
        return []

def download_poster(movie_id, cover_url, proxy=None):
    if not cover_url:
        return None
    session = create_session(proxy)
    headers = HEADERS.copy()
    headers['Referer'] = 'https://movie.douban.com/'
    try:
        resp = session.get(cover_url, headers=headers, stream=True, timeout=20)
        if resp.status_code == 200:
            ext = os.path.splitext(cover_url.split('?')[0])[1] or '.jpg'
            filename = f'{movie_id}{ext}'
            filepath = os.path.join(POSTER_DIR, filename)
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return filepath
        else:
            print(f"下载图片失败 {movie_id} 状态码 {resp.status_code}")
            return None
    except Exception as e:
        print(f"下载图片异常 {movie_id}: {e}")
        return None

# ====================== 主流程 ======================
def main():
    print('=' * 60)
    print('豆瓣电影数据爬虫（硬编码Cookie版）')
    print(f'目标数量: {TARGET_COUNT}')
    print('已使用您提供的 Cookie，预热中...')
    print('=' * 60)

    proxy = PROXY_LIST[0] if PROXY_LIST else None

    movie_list = collect_movie_ids(TARGET_COUNT, proxy)
    if not movie_list:
        print('❌ 未获取到任何电影，请检查 Cookie 是否过期。')
        return

    results = []
    for idx, movie in enumerate(movie_list, 1):
        movie_id = movie['id']
        print(f'\n处理第 {idx}/{len(movie_list)} 部: {movie["title"]} ({movie_id})')
        detail = parse_detail(movie_id, movie, proxy)
        if not detail:
            print('详情解析失败，跳过')
            time.sleep(random.uniform(*DELAY_RANGE))
            continue
        comments = get_hot_comments(movie_id, proxy)
        detail['hot_comments'] = comments
        cover_path = download_poster(movie_id, detail['cover'], proxy)
        detail['poster_local_path'] = cover_path
        results.append(detail)
        if idx % 50 == 0:
            with open('movies_data_temp.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f'已保存中间结果 {len(results)} 条')
        time.sleep(random.uniform(*DELAY_RANGE))

    if results:
        fieldnames = ['id','title','cover','rating','director','screenwriter','cast','genre','country','language','release_date','runtime','imdb','aka','summary','hot_comments','poster_local_path']
        with open('movies_data_final.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                item_copy = item.copy()
                if isinstance(item_copy.get('hot_comments'), list):
                    item_copy['hot_comments'] = '；'.join(item_copy['hot_comments'])
                writer.writerow(item_copy)
        print(f'\n✅ 爬取完成，共 {len(results)} 条数据，保存至 movies_data_final.csv')
    else:
        print('❌ 未获取到任何数据。')

if __name__ == '__main__':
    main()