"""
用户偏好综合推荐引擎 v2.0
基于 Spark MLlib + 图算法(NetworkX/Neo4j) + TF-IDF 的混合推荐

算法架构:
1. 用户信号加权聚合 → 偏好电影种子集合
2. Spark MLlib ALS 协同过滤 → 基于物品隐因子的相似推荐
3. 图算法推荐 → PageRank + Personalized PageRank + Louvain社区 + 共同邻居
4. TF-IDF 内容推荐 → 基于偏好类型的加权内容匹配
5. 多路融合排序 → 最终推荐列表

推荐权重:
  ALS协同过滤: 30%  |  图算法融合: 30%  |  TF-IDF内容: 25%  |  类型偏好: 15%

信号权重:
  评分(4-5星): 1.0 | 评分(3星): 0.5 | 收藏: 0.8 | 喜欢: 0.6 | 浏览: 0.2
"""
import json
import os
import re
import sys
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


class UserPreferenceRecommender:
    """用户偏好综合推荐器 (Spark MLlib + 图算法 + TF-IDF)"""

    def __init__(self, als_recommender, graph_recommender,
                 content_similarity, id_to_idx, idx_to_id, full_movie_data):
        self.als_rec = als_recommender
        self.graph_rec = graph_recommender
        self.content_similarity = content_similarity
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.full_movie_data = full_movie_data

        # 构建电影信息索引
        self.movie_info = {}
        self._build_movie_index()

        # 信号权重
        self.WEIGHTS = {
            'rating_high': 1.0,
            'rating_mid': 0.5,
            'rating_low': 0.1,
            'favorite': 0.8,
            'like': 0.6,
            'browse': 0.2,
        }

        # 推荐通道权重
        self.CHANNEL_WEIGHTS = {
            'als': 0.30,       # Spark MLlib ALS
            'graph': 0.30,     # 图算法 (PPR + CommonNeighbor + Community)
            'content': 0.25,   # TF-IDF 内容
            'genre': 0.15,     # 类型偏好
        }

    def _build_movie_index(self):
        """构建电影信息索引"""
        for m in self.full_movie_data:
            mid = str(m.get('id', ''))
            if not mid or not m.get('title'):
                continue
            genre_str = m.get('genre', '')
            genres = [g.strip() for g in genre_str.replace('/', ',').split(',') if g.strip()]
            self.movie_info[mid] = {
                'title': m.get('title', ''),
                'genres': genres,
                'rating': float(m.get('rating', 0)) if m.get('rating') else 0,
                'country': m.get('country', ''),
                'release_date': m.get('release_date', ''),
            }

    # ================================================================
    # 用户信号收集
    # ================================================================
    def get_user_signals(self, user_id):
        """收集用户所有交互信号（含喜欢的导演/演员）"""
        from database import (
            get_user_ratings, get_user_likes, get_user_favorites,
            get_browsing_history, get_liked_persons
        )

        signals = {
            'ratings': get_user_ratings(user_id, limit=100),
            'likes': get_user_likes(user_id, limit=100),
            'favorites': get_user_favorites(user_id, limit=100),
            'history': get_browsing_history(user_id, limit=100),
            'liked_directors': get_liked_persons(user_id, 'director', limit=30),
            'liked_actors': get_liked_persons(user_id, 'actor', limit=30),
        }
        return signals

    # ================================================================
    # 用户画像构建
    # ================================================================
    def build_user_profile(self, user_id):
        """构建用户偏好画像"""
        signals = self.get_user_signals(user_id)

        movie_weights = defaultdict(float)
        genre_counter = defaultdict(float)
        country_counter = defaultdict(float)
        total_rating = 0.0
        rating_count = 0

        # 1) 评分信号
        for r in signals['ratings']:
            mid = str(r['movie_id'])
            score = int(r['score'])
            if score >= 4:
                w = self.WEIGHTS['rating_high']
            elif score == 3:
                w = self.WEIGHTS['rating_mid']
            else:
                w = self.WEIGHTS['rating_low']
            movie_weights[mid] += w
            total_rating += score
            rating_count += 1
            if mid in self.movie_info:
                for g in self.movie_info[mid]['genres']:
                    genre_counter[g] += w
                c = self.movie_info[mid].get('country', '')
                if c:
                    country_counter[c] += w * 0.3

        # 2) 收藏信号
        for f in signals['favorites']:
            mid = str(f['movie_id'])
            w = self.WEIGHTS['favorite']
            movie_weights[mid] += w
            if mid in self.movie_info:
                for g in self.movie_info[mid]['genres']:
                    genre_counter[g] += w
                c = self.movie_info[mid].get('country', '')
                if c:
                    country_counter[c] += w * 0.3

        # 3) 喜欢信号
        for lk in signals['likes']:
            mid = str(lk['movie_id'])
            w = self.WEIGHTS['like']
            movie_weights[mid] += w
            if mid in self.movie_info:
                for g in self.movie_info[mid]['genres']:
                    genre_counter[g] += w
                c = self.movie_info[mid].get('country', '')
                if c:
                    country_counter[c] += w * 0.3

        # 4) 浏览历史
        for h in signals['history']:
            mid = str(h['movie_id'])
            w = self.WEIGHTS['browse']
            movie_weights[mid] += w
            if mid in self.movie_info:
                for g in self.movie_info[mid]['genres']:
                    genre_counter[g] += w * 0.5

        # 5) 喜欢的导演：找该导演的评分最高电影，加权
        DIRECTOR_WEIGHT = 0.7
        for p in signals.get('liked_directors', []):
            name = p['person_name']
            for m in self.full_movie_data:
                directors = [d.strip() for d in m.get('director', '').replace('/', ',').split(',') if d.strip()]
                if name in directors:
                    mid = str(m.get('id', ''))
                    if mid and mid in self.movie_info:
                        r = self.movie_info[mid]['rating']
                        w = DIRECTOR_WEIGHT * (r / 10.0) if r > 0 else DIRECTOR_WEIGHT * 0.5
                        movie_weights[mid] += w
                        for g in self.movie_info[mid]['genres']:
                            genre_counter[g] += w * 0.4

        # 6) 喜欢的演员：找该演员参演的评分最高电影，加权
        ACTOR_WEIGHT = 0.5
        for p in signals.get('liked_actors', []):
            name = p['person_name']
            for m in self.full_movie_data:
                actors = [a.strip() for a in m.get('cast', '').replace('/', ',').split(',') if a.strip()]
                if name in actors:
                    mid = str(m.get('id', ''))
                    if mid and mid in self.movie_info:
                        r = self.movie_info[mid]['rating']
                        w = ACTOR_WEIGHT * (r / 10.0) if r > 0 else ACTOR_WEIGHT * 0.5
                        movie_weights[mid] += w
                        for g in self.movie_info[mid]['genres']:
                            genre_counter[g] += w * 0.3

        # 归一化
        max_w = max(movie_weights.values()) if movie_weights else 1.0
        movie_weights = {k: v / max_w for k, v in movie_weights.items()}

        sorted_genres = sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)
        sorted_countries = sorted(country_counter.items(), key=lambda x: x[1], reverse=True)

        return {
            'preferred_movies': dict(movie_weights),
            'genre_prefs': sorted_genres[:10],
            'country_prefs': sorted_countries[:5],
            'avg_rating': round(total_rating / rating_count, 1) if rating_count > 0 else 0,
            'total_interactions': (
                len(signals['ratings']) + len(signals['favorites']) +
                len(signals['likes']) + len(signals['history'])
            ),
            'signal_summary': {
                'ratings': len(signals['ratings']),
                'favorites': len(signals['favorites']),
                'likes': len(signals['likes']),
                'history': len(signals['history']),
                'liked_directors': len(signals.get('liked_directors', [])),
                'liked_actors': len(signals.get('liked_actors', [])),
            }
        }

    # ================================================================
    # 混合推荐算法 (四路融合)
    # ================================================================
    def recommend(self, user_id, top_n=24):
        """综合推荐: ALS + 图算法 + TF-IDF + 类型偏好"""
        profile = self.build_user_profile(user_id)

        if profile['total_interactions'] == 0:
            return self._cold_start_recommend(top_n), profile

        # 确保图已构建
        if not self.graph_rec.graph_built:
            self.graph_rec.build_graph()

        # 四路推荐分数
        als_scores = self._als_scores(profile)
        graph_scores = self._graph_scores(profile)
        content_scores = self._content_scores(profile)
        genre_scores = self._genre_match_scores(profile)

        # 收集所有候选电影ID
        all_movie_ids = set()
        all_movie_ids.update(als_scores.keys())
        all_movie_ids.update(graph_scores.keys())
        all_movie_ids.update(content_scores.keys())
        all_movie_ids.update(genre_scores.keys())

        # 排除用户已交互的电影
        exclude_ids = set(profile['preferred_movies'].keys())

        # 多路融合计分
        final_scores = []
        for mid in all_movie_ids:
            if mid in exclude_ids:
                continue
            if mid not in self.movie_info:
                continue

            score = (
                als_scores.get(mid, 0) * self.CHANNEL_WEIGHTS['als'] +
                graph_scores.get(mid, 0) * self.CHANNEL_WEIGHTS['graph'] +
                content_scores.get(mid, 0) * self.CHANNEL_WEIGHTS['content'] +
                genre_scores.get(mid, 0) * self.CHANNEL_WEIGHTS['genre']
            )

            # 全局评分加成
            global_rating = self.movie_info[mid]['rating']
            if global_rating > 0:
                score += (global_rating / 10.0) * 0.05

            if score > 0:
                final_scores.append((mid, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for mid, score in final_scores[:top_n]:
            info = self.movie_info[mid]
            als_c = als_scores.get(mid, 0)
            gr_c = graph_scores.get(mid, 0)
            ct_c = content_scores.get(mid, 0)
            gn_c = genre_scores.get(mid, 0)

            reasons = []
            if als_c > 0.08:
                reasons.append('ALS协同过滤')
            if gr_c > 0.05:
                reasons.append('图算法')
            if ct_c > 0.05:
                reasons.append('内容相似')
            if gn_c > 0.08:
                reasons.append('类型偏好')

            results.append({
                'id': mid,
                'title': info['title'],
                'genres': info['genres'],
                'rating': info['rating'],
                'country': info['country'],
                'release_date': info['release_date'],
                'score': round(score, 4),
                'als_score': round(als_c, 4),
                'graph_score': round(gr_c, 4),
                'content_score': round(ct_c, 4),
                'genre_score': round(gn_c, 4),
                'reasons': reasons,
            })

        return results, profile

    # ================================================================
    # ALS 协同过滤得分 (Spark MLlib)
    # ================================================================
    def _als_scores(self, profile):
        """基于ALS物品因子: 用偏好电影种子找相似电影"""
        if not self.als_rec.model_trained or self.als_rec.item_factors is None:
            return {}

        item_factors = self.als_rec.item_factors
        scores = defaultdict(float)

        top_movies = sorted(profile['preferred_movies'].items(),
                           key=lambda x: x[1], reverse=True)[:15]

        for mid, weight in top_movies:
            idx = self.als_rec.movie_id_to_idx.get(mid)
            if idx is None:
                continue
            target_vec = item_factors[idx]
            target_norm = np.linalg.norm(target_vec)
            if target_norm == 0:
                continue

            norms = np.linalg.norm(item_factors, axis=1)
            similarities = item_factors @ target_vec / (norms * target_norm + 1e-8)

            for i, sim in enumerate(similarities):
                if i == idx:
                    continue
                if sim > 0.1:
                    m_id = self.als_rec.idx_to_movie_id.get(i)
                    if m_id:
                        scores[m_id] += sim * weight

        max_s = max(scores.values()) if scores else 1.0
        return {k: v / max_s for k, v in scores.items()}

    # ================================================================
    # 图算法推荐得分 (NetworkX + PageRank + PPR + Community + CommonNeighbor)
    # ================================================================
    def _graph_scores(self, profile):
        """
        图算法融合得分:
        对每部偏好电影种子, 运行混合图推荐, 加权聚合
        使用: Personalized PageRank + 共同邻居 + Louvain社区
        """
        if not self.graph_rec.graph_built:
            try:
                self.graph_rec.build_graph()
            except Exception:
                return {}

        scores = defaultdict(float)

        top_movies = sorted(profile['preferred_movies'].items(),
                           key=lambda x: x[1], reverse=True)[:10]

        for mid, weight in top_movies:
            # PPR (权重最高: 图算法核心)
            ppr_recs = self.graph_rec.personalized_pagerank_recommend(mid, top_n=20)
            for r in ppr_recs:
                scores[r['id']] += r.get('similarity', 0) * weight * 0.45

            # 共同邻居
            cn_recs = self.graph_rec.common_neighbor_recommend(mid, top_n=20)
            for r in cn_recs:
                scores[r['id']] += r.get('similarity', 0) * weight * 0.35

            # 社区推荐
            comm_recs = self.graph_rec.community_recommend(mid, top_n=20)
            for r in comm_recs:
                scores[r['id']] += (r.get('similarity', 0) / 10.0) * weight * 0.20

        # 全局PageRank加成: 偏好电影类型相关的高PageRank电影
        top_genres = [g for g, _ in profile['genre_prefs'][:3]]
        pr_movies = self.graph_rec.pagerank_recommend(top_n=100)
        for r in pr_movies:
            m_genres = self.movie_info.get(r['id'], {}).get('genres', [])
            if any(g in m_genres for g in top_genres):
                scores[r['id']] += r.get('pagerank_score', 0) * 100 * 0.10

        max_s = max(scores.values()) if scores else 1.0
        return {k: v / max_s for k, v in scores.items()}

    # ================================================================
    # TF-IDF 内容相似度得分
    # ================================================================
    def _content_scores(self, profile):
        """基于TF-IDF内容相似度, 偏好电影加权扩散"""
        if self.content_similarity is None:
            return {}

        scores = defaultdict(float)

        top_movies = sorted(profile['preferred_movies'].items(),
                           key=lambda x: x[1], reverse=True)[:10]

        for mid, weight in top_movies:
            idx = self.id_to_idx.get(mid)
            if idx is None:
                continue
            sim_vector = self.content_similarity[idx]
            for i, sim in enumerate(sim_vector):
                if sim > 0.15:
                    m_id = self.idx_to_id.get(i)
                    if m_id:
                        scores[m_id] += sim * weight

        max_s = max(scores.values()) if scores else 1.0
        return {k: v / max_s for k, v in scores.items()}

    # ================================================================
    # 类型偏好匹配得分
    # ================================================================
    def _genre_match_scores(self, profile):
        """基于用户类型偏好的匹配度"""
        if not profile['genre_prefs']:
            return {}

        genre_weights = {g: s for g, s in profile['genre_prefs']}
        max_gw = max(genre_weights.values()) if genre_weights else 1.0
        genre_weights = {g: s / max_gw for g, s in genre_weights.items()}

        scores = {}
        for mid, info in self.movie_info.items():
            score = 0.0
            for g in info['genres']:
                score += genre_weights.get(g, 0)
            if score > 0:
                scores[mid] = score / max(1, len(info['genres']))

        return scores

    # ================================================================
    # 基于单部电影推荐相似电影
    # ================================================================
    def recommend_for_movie(self, user_id, movie_id, top_n=12):
        """基于单部电影的相似推荐（用于搜索推荐）"""
        mid = str(movie_id)
        results = []

        # ALS 相似
        if self.als_rec.model_trained and self.als_rec.item_factors is not None:
            idx = self.als_rec.movie_id_to_idx.get(mid)
            if idx is not None:
                target = self.als_rec.item_factors[idx]
                norms = np.linalg.norm(self.als_rec.item_factors, axis=1)
                target_norm = np.linalg.norm(target)
                if target_norm > 0:
                    sims = self.als_rec.item_factors @ target / (norms * target_norm + 1e-8)
                    for i, s in enumerate(sims):
                        if i != idx and s > 0.15:
                            m_id = self.als_rec.idx_to_movie_id.get(i)
                            if m_id and m_id in self.movie_info:
                                results.append({'id': m_id, 'score': float(s)})

        # 图算法
        if self.graph_rec.graph_built:
            for r in self.graph_rec.personalized_pagerank_recommend(mid, top_n=10):
                results.append({'id': r['id'], 'score': r.get('similarity', 0) * 0.8})

        # 按分数排序去重
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x['score'], reverse=True):
            if r['id'] not in seen and r['id'] != mid and r['id'] in self.movie_info:
                seen.add(r['id'])
                info = self.movie_info[r['id']]
                unique.append({
                    'id': r['id'], 'title': info['title'],
                    'genres': info['genres'], 'rating': info['rating'],
                    'country': info['country'], 'release_date': info['release_date'],
                    'score': round(r['score'], 4),
                })
            if len(unique) >= top_n:
                break

        return unique, None

    # ================================================================
    # 冷启动推荐
    # ================================================================
    def _cold_start_recommend(self, top_n=24):
        """冷启动: 高分 + 高PageRank"""
        # 如果图已构建, 用PageRank
        if self.graph_rec.graph_built:
            return self.graph_rec.pagerank_recommend(top_n=top_n)

        # 否则用高分热门
        candidates = []
        for mid, info in self.movie_info.items():
            if info['rating'] >= 7.0:
                candidates.append({
                    'id': mid, 'title': info['title'],
                    'genres': info['genres'], 'rating': info['rating'],
                    'country': info['country'], 'release_date': info['release_date'],
                    'score': info['rating'] / 10.0,
                    'als_score': 0, 'graph_score': 0, 'content_score': 0, 'genre_score': 0,
                    'reasons': ['热门高分推荐'],
                })
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]

    # ================================================================
    # 用户画像分析
    # ================================================================
    def analyze_profile(self, user_id):
        """全面分析用户偏好"""
        profile = self.build_user_profile(user_id)

        top_genres = profile['genre_prefs'][:5]

        year_counter = defaultdict(int)
        for mid in profile['preferred_movies']:
            info = self.movie_info.get(mid)
            if info and info.get('release_date'):
                match = re.search(r'(\d{4})', info['release_date'])
                if match:
                    yr = int(match.group(1))
                    decade = (yr // 10) * 10
                    year_counter[decade] += 1
        top_decades = sorted(year_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'profile': profile,
            'top_genres': top_genres,
            'top_decades': [(f'{d}s', c) for d, c in top_decades],
            'top_countries': profile['country_prefs'],
        }
