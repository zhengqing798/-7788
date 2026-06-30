"""
数据可视化分析模块
多维度电影数据分析
"""
import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


class MovieAnalyzer:
    """电影数据分析器"""

    def __init__(self, data_path=None):
        if data_path is None:
            # 使用脚本所在目录的上级目录（即项目根目录）来定位数据文件
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'movies_data_temp(1).json')
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.movies = []
        for m in raw_data:
            if not m.get('title') or not m.get('id'):
                continue
            try:
                rating = float(m.get('rating', 0))
            except (ValueError, TypeError):
                rating = 0.0

            genre_str = m.get('genre', '')
            genres = [g.strip() for g in genre_str.replace('/', ',').split(',') if g.strip()]

            country_str = m.get('country', '')
            countries = [c.strip() for c in country_str.replace('/', ',').split(',') if c.strip()]

            # Parse year
            date_str = m.get('release_date', '')
            year = None
            if date_str:
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    year = int(year_match.group(1))

            # Parse runtime
            runtime_str = m.get('runtime', '')
            runtime_num = 0
            if runtime_str:
                match = re.search(r'(\d+)', str(runtime_str))
                if match:
                    runtime_num = int(match.group(1))

            # Parse director count
            directors = [d.strip() for d in m.get('director', '').replace('/', ',').split(',') if d.strip()]

            # Parse actors
            actors = [a.strip() for a in m.get('cast', '').replace('/', ',').split(',') if a.strip()]
            top_actors = actors[:5] if len(actors) >= 5 else actors

            self.movies.append({
                'id': str(m['id']),
                'title': m['title'].strip(),
                'rating': rating,
                'genres': genres,
                'countries': countries,
                'year': year,
                'runtime': runtime_num,
                'directors': directors,
                'top_actors': top_actors,
                'all_actors': actors,
                'language': m.get('language', ''),
            })

    # ================================================================
    # 1. 评分分布分析
    # ================================================================
    def rating_distribution(self):
        """评分分布 - 直方图"""
        ratings = [m['rating'] for m in self.movies if m['rating'] > 0]
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        hist, _ = np.histogram(ratings, bins=bins)
        labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
        return {
            'labels': labels,
            'values': hist.tolist(),
            'avg': round(np.mean(ratings), 2),
            'median': round(np.median(ratings), 2),
            'std': round(np.std(ratings), 2),
            'min': min(ratings),
            'max': max(ratings),
        }

    # ================================================================
    # 2. 类型分布分析
    # ================================================================
    def genre_distribution(self, top_n=20):
        """电影类型分布"""
        genre_counter = Counter()
        genre_avg_rating = defaultdict(list)

        for m in self.movies:
            for g in m['genres']:
                genre_counter[g] += 1
                if m['rating'] > 0:
                    genre_avg_rating[g].append(m['rating'])

        top_genres = genre_counter.most_common(top_n)
        return {
            'labels': [g for g, _ in top_genres],
            'values': [c for _, c in top_genres],
            'avg_ratings': [round(np.mean(genre_avg_rating[g]), 2) for g, _ in top_genres],
        }

    # ================================================================
    # 3. 国家/地区分布
    # ================================================================
    def country_distribution(self, top_n=15):
        """制片国家/地区分布"""
        country_counter = Counter()
        country_rating = defaultdict(list)

        for m in self.movies:
            for c in m['countries']:
                country_counter[c] += 1
                if m['rating'] > 0:
                    country_rating[c].append(m['rating'])

        top_countries = country_counter.most_common(top_n)
        return {
            'labels': [c for c, _ in top_countries],
            'values': [cnt for _, cnt in top_countries],
            'avg_ratings': [round(np.mean(country_rating[c]), 2) for c, _ in top_countries],
        }

    # ================================================================
    # 4. 年份趋势分析
    # ================================================================
    def year_trend(self, start_year=1950, end_year=2026):
        """年份趋势：每年电影数量和平均评分"""
        year_data = defaultdict(lambda: {'count': 0, 'ratings': []})

        for m in self.movies:
            y = m['year']
            if y and start_year <= y <= end_year:
                year_data[y]['count'] += 1
                if m['rating'] > 0:
                    year_data[y]['ratings'].append(m['rating'])

        years = sorted(year_data.keys())
        return {
            'years': years,
            'counts': [year_data[y]['count'] for y in years],
            'avg_ratings': [
                round(np.mean(year_data[y]['ratings']), 2) if year_data[y]['ratings'] else 0
                for y in years
            ],
        }

    # ================================================================
    # 5. 片长分布分析
    # ================================================================
    def runtime_distribution(self):
        """片长分布"""
        runtimes = [m['runtime'] for m in self.movies if m['runtime'] > 0 and m['runtime'] < 400]
        if not runtimes:
            return {'labels': [], 'values': []}

        # 30分钟为一个区间
        max_rt = max(runtimes)
        bins = list(range(0, int(max_rt) + 31, 30))
        hist, _ = np.histogram(runtimes, bins=bins)
        labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
        return {
            'labels': labels,
            'values': hist.tolist(),
            'avg': round(np.mean(runtimes), 1),
            'min': min(runtimes),
            'max': max(runtimes),
        }

    # ================================================================
    # 6. 评分 vs 片长散点图
    # ================================================================
    def rating_vs_runtime(self):
        """评分与片长的关系"""
        points = []
        for m in self.movies:
            if m['rating'] > 0 and m['runtime'] > 0 and m['runtime'] < 400:
                points.append({
                    'x': m['runtime'],
                    'y': m['rating'],
                    'title': m['title'],
                    'genre': m['genres'][0] if m['genres'] else '',
                })
        # Sample to avoid too many points
        if len(points) > 1000:
            import random
            random.seed(42)
            points = random.sample(points, 1000)
        return {'points': points}

    # ================================================================
    # 7. 语言分布
    # ================================================================
    def language_distribution(self, top_n=15):
        """语言分布"""
        lang_counter = Counter()
        for m in self.movies:
            langs = [l.strip() for l in m['language'].replace('/', ',').split(',') if l.strip()]
            for lang in langs:
                lang_counter[lang] += 1

        top_langs = lang_counter.most_common(top_n)
        return {
            'labels': [l for l, _ in top_langs],
            'values': [c for _, c in top_langs],
        }

    # ================================================================
    # 8. 导演作品数量排名
    # ================================================================
    def director_ranking(self, top_n=20):
        """导演作品数量排名"""
        director_counter = Counter()
        director_rating = defaultdict(list)

        for m in self.movies:
            for d in m['directors']:
                if d:
                    director_counter[d] += 1
                    if m['rating'] > 0:
                        director_rating[d].append(m['rating'])

        top_directors = director_counter.most_common(top_n)
        return {
            'labels': [d for d, _ in top_directors],
            'values': [c for _, c in top_directors],
            'avg_ratings': [round(np.mean(director_rating[d]), 2) for d, _ in top_directors],
        }

    # ================================================================
    # 9. 演员参演数量排名
    # ================================================================
    def actor_ranking(self, top_n=20):
        """演员参演数量排名"""
        actor_counter = Counter()
        actor_rating = defaultdict(list)

        for m in self.movies:
            for a in m['top_actors']:
                if a:
                    actor_counter[a] += 1
                    if m['rating'] > 0:
                        actor_rating[a].append(m['rating'])

        top_actors = actor_counter.most_common(top_n)
        return {
            'labels': [a for a, _ in top_actors],
            'values': [c for _, c in top_actors],
            'avg_ratings': [round(np.mean(actor_rating[a]), 2) for a, _ in top_actors],
        }

    # ================================================================
    # 10. 高分电影Top榜单 (按类型)
    # ================================================================
    def top_rated_by_genre(self, genre, top_n=10):
        """某个类型中评分最高的电影"""
        candidates = [m for m in self.movies if genre in m['genres'] and m['rating'] > 0]
        candidates.sort(key=lambda x: x['rating'], reverse=True)
        return [
            {
                'id': m['id'],
                'title': m['title'],
                'rating': m['rating'],
                'year': m['year'],
                'genres': m['genres'],
            }
            for m in candidates[:top_n]
        ]

    # ================================================================
    # 11. 全部统计数据汇总
    # ================================================================
    def full_report(self):
        """生成完整的分析报告"""
        return {
            'rating_distribution': self.rating_distribution(),
            'genre_distribution': self.genre_distribution(),
            'country_distribution': self.country_distribution(),
            'year_trend': self.year_trend(),
            'runtime_distribution': self.runtime_distribution(),
            'rating_vs_runtime': self.rating_vs_runtime(),
            'language_distribution': self.language_distribution(),
            'director_ranking': self.director_ranking(),
            'actor_ranking': self.actor_ranking(),
        }

    def summary_stats(self):
        """基本统计数据"""
        ratings = [m['rating'] for m in self.movies if m['rating'] > 0]
        runtimes = [m['runtime'] for m in self.movies if m['runtime'] > 0]
        years = [m['year'] for m in self.movies if m['year']]

        all_genres = set()
        for m in self.movies:
            all_genres.update(m['genres'])

        all_countries = set()
        for m in self.movies:
            all_countries.update(m['countries'])

        return {
            'total_movies': len(self.movies),
            'avg_rating': round(np.mean(ratings), 2),
            'median_rating': round(np.median(ratings), 2),
            'max_rating': max(ratings),
            'min_rating': min(ratings),
            'std_rating': round(np.std(ratings), 2),
            'avg_runtime': round(np.mean(runtimes), 1),
            'year_range': f'{min(years)}-{max(years)}',
            'genre_count': len(all_genres),
            'country_count': len(all_countries),
        }
