"""
图算法推荐引擎
结合NetworkX图分析和Neo4j图数据库进行电影推荐

图算法:
- PageRank: 找出最重要的电影节点
- Louvain社区检测: 发现电影聚类
- Personalized PageRank: 基于种子电影的个性化推荐
- 共同邻居相似度: 基于共享演员/导演的推荐
- 最短路径: 电影之间的关系链
"""
import json
import os
import pickle
import numpy as np
from collections import defaultdict, Counter


class GraphRecommender:
    """
    基于图算法的电影推荐器
    支持NetworkX (内置) 和 Neo4j (可选)
    """

    def __init__(self, data_path=None, cache_dir='cache'):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'movies_data_temp(1).json')
        self.data_path = data_path
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cache_dir) if not os.path.isabs(cache_dir) else cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.movies = []
        self.movie_by_id = {}
        self.graph = None
        self.pagerank_scores = {}
        self.communities = {}
        self.graph_built = False

        self._load_data()
        self._check_neo4j()

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for m in raw:
            if not m.get('title') or not m.get('id'):
                continue
            try:
                rating = float(m.get('rating', 0))
            except (ValueError, TypeError):
                rating = 0.0

            genres = [g.strip() for g in m.get('genre', '').replace('/', ',').split(',') if g.strip()]
            directors = [d.strip() for d in m.get('director', '').replace('/', ',').split(',') if d.strip()]
            actors = [a.strip() for a in m.get('cast', '').replace('/', ',').split(',') if a.strip()][:8]
            countries = [c.strip() for c in m.get('country', '').replace('/', ',').split(',') if c.strip()]

            self.movies.append({
                'id': str(m['id']),
                'title': m['title'].strip(),
                'cover': m.get('cover', ''),
                'rating': rating,
                'genres': genres,
                'directors': directors,
                'actors': actors,
                'countries': countries,
                'release_date': m.get('release_date', ''),
                'poster_local_path': m.get('poster_local_path', ''),
            })
            self.movie_by_id[str(m['id'])] = self.movies[-1]

    def _check_neo4j(self):
        """检查Neo4j连接"""
        self.neo4j_available = False
        try:
            from py2neo import Graph
            try:
                self.neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
                self.neo4j_graph.run("MATCH (n) RETURN count(n) LIMIT 1")
                self.neo4j_available = True
                print("Neo4j连接成功")
            except Exception:
                print("Neo4j服务不可用，使用NetworkX内存图")
        except ImportError:
            print("py2neo不可用，使用NetworkX内存图")

    # ================================================================
    # 构建图
    # ================================================================
    def build_graph(self, force_rebuild=False):
        """构建电影知识图谱"""
        cache_file = os.path.join(self.cache_dir, 'movie_graph.pkl')
        if os.path.exists(cache_file) and not force_rebuild:
            print("加载缓存的图...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.pagerank_scores = data['pagerank']
                self.communities = data.get('communities', {})
                self.graph_built = True
            return

        print("构建电影知识图谱...")
        import networkx as nx

        G = nx.Graph()

        # 电影节点
        for m in self.movies:
            G.add_node(m['id'], type='Movie', title=m['title'],
                       rating=m['rating'], genres=m['genres'])

        # 类型节点和边
        genre_nodes = set()
        for m in self.movies:
            for g in m['genres']:
                genre_node = f"genre:{g}"
                if genre_node not in genre_nodes:
                    G.add_node(genre_node, type='Genre', name=g)
                    genre_nodes.add(genre_node)
                # 电影-类型边
                weight = 1.0
                if m['rating'] > 7.5:
                    weight = 2.0
                G.add_edge(m['id'], genre_node, relation='BELONGS_TO', weight=weight)

        # 导演节点和边
        director_nodes = set()
        for m in self.movies:
            for d in m['directors']:
                if d:
                    d_node = f"director:{d}"
                    if d_node not in director_nodes:
                        G.add_node(d_node, type='Director', name=d)
                        director_nodes.add(d_node)
                    G.add_edge(m['id'], d_node, relation='DIRECTED_BY', weight=1.5)

        # 演员节点和边
        actor_nodes = set()
        for m in self.movies:
            for a in m['actors'][:5]:
                if a:
                    a_node = f"actor:{a}"
                    if a_node not in actor_nodes:
                        G.add_node(a_node, type='Actor', name=a)
                        actor_nodes.add(a_node)
                    G.add_edge(m['id'], a_node, relation='ACTED_IN', weight=1.0)

        # 国家节点和边
        country_nodes = set()
        for m in self.movies:
            for c in m['countries']:
                if c:
                    c_node = f"country:{c}"
                    if c_node not in country_nodes:
                        G.add_node(c_node, type='Country', name=c)
                        country_nodes.add(c_node)
                    G.add_edge(m['id'], c_node, relation='FROM_COUNTRY', weight=0.5)

        self.graph = G
        print(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

        # 计算PageRank
        print("计算PageRank...")
        self.pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)

        # 社区检测 (仅对电影节点)
        print("检测社区...")
        try:
            import community as community_louvain
            movie_nodes = [m['id'] for m in self.movies]
            subgraph = G.subgraph([n for n in movie_nodes if n in G.nodes])
            partition = community_louvain.best_partition(subgraph.to_undirected())
            self.communities = {}
            for node, comm_id in partition.items():
                self.communities[node] = comm_id
            print(f"发现 {len(set(partition.values()))} 个社区")
        except ImportError:
            # 备用：基于类型的简单社区
            self.communities = {}
            for m in self.movies:
                if m['genres']:
                    self.communities[m['id']] = hash(m['genres'][0]) % 50

        self.graph_built = True

        # 缓存
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'graph': G,
                'pagerank': self.pagerank_scores,
                'communities': self.communities,
            }, f)

    # ================================================================
    # PageRank推荐 - 最重要/最相关的电影
    # ================================================================
    def pagerank_recommend(self, top_n=20, genre=None):
        """基于PageRank的电影推荐 - 发现最重要/最具影响力的电影"""
        if not self.graph_built:
            self.build_graph()

        movie_scores = []
        for m in self.movies:
            movie_id = m['id']
            pr = self.pagerank_scores.get(movie_id, 0)

            # 考虑评分加成
            rating_bonus = m['rating'] / 10.0 if m['rating'] > 0 else 0
            combined = pr * 0.7 + rating_bonus * 0.3

            if genre and genre not in m['genres']:
                continue

            movie_scores.append((movie_id, combined, pr))

        movie_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score, pr in movie_scores[:top_n]:
            m = self.movie_by_id.get(mid)
            if m:
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': ', '.join(m['genres'][:3]),
                    'pagerank_score': round(float(pr), 6),
                    'combined_score': round(float(score), 4),
                    'cover': m['cover'],
                    'poster_local_path': m['poster_local_path'],
                    'method': 'PageRank图算法'
                })
        return results

    # ================================================================
    # 共同邻居相似度 - 基于共享关系的推荐
    # ================================================================
    def common_neighbor_recommend(self, movie_id, top_n=12):
        """基于共同邻居的推荐（共享演员/导演/类型越多越相似）"""
        if not self.graph_built:
            self.build_graph()

        import networkx as nx

        target = str(movie_id)
        if target not in self.graph:
            return []

        neighbors = set(self.graph.neighbors(target))
        if not neighbors:
            return []

        scores = {}
        for neighbor in neighbors:
            for candidate in self.graph.neighbors(neighbor):
                if candidate == target:
                    continue
                if not candidate.startswith('genre:') and not candidate.startswith('director:') \
                   and not candidate.startswith('actor:') and not candidate.startswith('country:'):
                    # candidate是电影节点
                    # Jaccard系数
                    candidate_neighbors = set(self.graph.neighbors(candidate))
                    if candidate_neighbors:
                        jaccard = len(neighbors & candidate_neighbors) / len(neighbors | candidate_neighbors)
                        # 加权：类型匹配多加分
                        target_m = self.movie_by_id.get(target, {})
                        cand_m = self.movie_by_id.get(candidate, {})
                        genre_overlap = len(
                            set(target_m.get('genres', [])) & set(cand_m.get('genres', []))
                        ) if target_m and cand_m else 0
                        weight = jaccard * 0.6 + (genre_overlap / max(1, len(target_m.get('genres', [1])))) * 0.4
                        scores[candidate] = max(scores.get(candidate, 0), weight)

        sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for mid, score in sorted_movies:
            m = self.movie_by_id.get(mid)
            if m:
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': ', '.join(m['genres'][:3]),
                    'similarity': round(float(score), 3),
                    'cover': m['cover'],
                    'poster_local_path': m['poster_local_path'],
                    'method': '图共同邻居相似度'
                })
        return results

    # ================================================================
    # Personalized PageRank (PPR) - 个性化图推荐
    # ================================================================
    def personalized_pagerank_recommend(self, movie_id, top_n=12):
        """基于Personalized PageRank的个性化推荐"""
        if not self.graph_built:
            self.build_graph()

        import networkx as nx

        target = str(movie_id)
        if target not in self.graph:
            return []

        # 设置个性化向量（种子节点权重高）
        personalization = {target: 0.5}
        # 加入种子节点的邻居
        neighbors = list(self.graph.neighbors(target))
        for n in neighbors[:20]:
            personalization[n] = 0.5 / min(20, len(neighbors))

        # 计算PPR
        try:
            ppr = nx.pagerank(self.graph, alpha=0.85, personalization=personalization,
                            max_iter=100, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            ppr = self.pagerank_scores

        # 提取电影节点
        movie_scores = {}
        for node, score in ppr.items():
            if not node.startswith(('genre:', 'director:', 'actor:', 'country:')):
                if node != target and node in self.movie_by_id:
                    m = self.movie_by_id[node]
                    # 加分：同类型
                    target_m = self.movie_by_id.get(target)
                    if target_m:
                        overlap = len(set(target_m['genres']) & set(m['genres']))
                        score *= (1 + 0.2 * overlap)
                    movie_scores[node] = score

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for mid, score in sorted_movies:
            m = self.movie_by_id.get(mid)
            if m:
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': ', '.join(m['genres'][:3]),
                    'similarity': round(float(score), 5),
                    'cover': m['cover'],
                    'poster_local_path': m['poster_local_path'],
                    'method': 'Personalized PageRank'
                })
        return results

    # ================================================================
    # 社区推荐 - 同社区电影
    # ================================================================
    def community_recommend(self, movie_id, top_n=12):
        """基于Louvain社区检测的同社区电影推荐"""
        if not self.graph_built:
            self.build_graph()

        target = str(movie_id)
        if target not in self.communities:
            return []

        target_community = self.communities[target]

        # 同社区电影，按评分排序
        community_movies = []
        for mid, comm in self.communities.items():
            if comm == target_community and mid != target:
                m = self.movie_by_id.get(mid)
                if m:
                    # 离种子电影越近越好
                    try:
                        distance = len(list(self.graph.neighbors(target)) & set(self.graph.neighbors(mid)))
                    except:
                        distance = 0
                    score = m['rating'] * 0.7 + distance * 0.3
                    community_movies.append((mid, score, m))

        community_movies.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score, m in community_movies[:top_n]:
            results.append({
                'id': mid,
                'title': m['title'],
                'rating': m['rating'],
                'genre': ', '.join(m['genres'][:3]),
                'similarity': round(float(score), 3),
                'community_id': target_community,
                'cover': m['cover'],
                'poster_local_path': m['poster_local_path'],
                'method': f'Louvain社区检测(社区#{target_community})'
            })
        return results

    # ================================================================
    # 最短路径 - 电影之间的关系链
    # ================================================================
    def movie_path(self, movie_id1, movie_id2):
        """查找两部电影之间的最短关系路径"""
        if not self.graph_built:
            self.build_graph()

        import networkx as nx

        source = str(movie_id1)
        target = str(movie_id2)

        if source not in self.graph or target not in self.graph:
            return {'error': '电影未在图中找到'}

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            path_info = []
            for i, node in enumerate(path):
                if node.startswith('genre:'):
                    path_info.append({'type': 'genre', 'name': node.replace('genre:', '')})
                elif node.startswith('director:'):
                    path_info.append({'type': 'director', 'name': node.replace('director:', '')})
                elif node.startswith('actor:'):
                    path_info.append({'type': 'actor', 'name': node.replace('actor:', '')})
                elif node.startswith('country:'):
                    path_info.append({'type': 'country', 'name': node.replace('country:', '')})
                else:
                    m = self.movie_by_id.get(node, {})
                    path_info.append({
                        'type': 'movie',
                        'name': m.get('title', node),
                        'id': node
                    })
            return {'path': path_info, 'length': len(path) - 1}
        except nx.NetworkXNoPath:
            return {'error': '两部电影之间没有路径连接'}
        except nx.NodeNotFound:
            return {'error': '节点未在图中找到'}

    # ================================================================
    # 图统计信息
    # ================================================================
    def graph_stats(self):
        """图算法统计"""
        if not self.graph_built:
            self.build_graph()

        import networkx as nx

        movie_nodes = [n for n in self.graph.nodes if not str(n).startswith(('genre:', 'director:', 'actor:', 'country:'))]
        genre_nodes = [n for n in self.graph.nodes if str(n).startswith('genre:')]
        director_nodes = [n for n in self.graph.nodes if str(n).startswith('director:')]
        actor_nodes = [n for n in self.graph.nodes if str(n).startswith('actor:')]

        # 度数最高的电影
        degrees = sorted(self.graph.degree(movie_nodes), key=lambda x: x[1], reverse=True)
        top_connected = []
        for node, deg in degrees[:20]:
            m = self.movie_by_id.get(node)
            if m:
                top_connected.append({
                    'id': node,
                    'title': m['title'],
                    'degree': deg,
                    'rating': m['rating'],
                    'genre': ', '.join(m['genres'][:3])
                })

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'movie_nodes': len(movie_nodes),
            'genre_nodes': len(genre_nodes),
            'director_nodes': len(director_nodes),
            'actor_nodes': len(actor_nodes),
            'avg_degree': round(sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(), 2),
            'density': round(nx.density(self.graph), 6),
            'num_communities': len(set(self.communities.values())),
            'top_connected_movies': top_connected,
            'top_pagerank': [
                {
                    'title': self.movie_by_id.get(n, {}).get('title', n),
                    'pagerank': round(float(s), 6)
                }
                for n, s in sorted(self.pagerank_scores.items(),
                                  key=lambda x: x[1], reverse=True)[:10]
                if not str(n).startswith(('genre:', 'director:', 'actor:', 'country:'))
            ][:10],
        }

    # ================================================================
    # Neo4j 查询接口 (当Neo4j可用时)
    # ================================================================
    def neo4j_recommend_by_genre_director(self, movie_id, top_n=10):
        """Neo4j Cypher查询: 同导演+同类型电影推荐"""
        if not self.neo4j_available:
            return None  # 回退到NetworkX

        try:
            query = """
            MATCH (m:Movie {id: $movie_id})-[:DIRECTED_BY]->(d:Director)<-[:DIRECTED_BY]-(rec:Movie)
            WHERE rec.id <> $movie_id
            WITH rec, d, COUNT(DISTINCT d) as shared_directors
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(rec)
            WITH rec, shared_directors, COUNT(DISTINCT g) as shared_genres
            RETURN rec.id as id, rec.title as title, rec.rating as rating,
                   shared_directors, shared_genres,
                   (shared_directors * 3 + shared_genres * 1) as score
            ORDER BY score DESC
            LIMIT $top_n
            """
            result = self.neo4j_graph.run(query, movie_id=str(movie_id), top_n=top_n).data()
            return result
        except Exception as e:
            print(f"Neo4j查询出错: {e}")
            return None

    # ================================================================
    # 综合图推荐 (融合多种图算法)
    # ================================================================
    def hybrid_graph_recommend(self, movie_id, top_n=12):
        """融合多种图算法的综合推荐"""
        # 获取各种推荐
        cn_recs = self.common_neighbor_recommend(movie_id, top_n=15)
        ppr_recs = self.personalized_pagerank_recommend(movie_id, top_n=15)
        community_recs = self.community_recommend(movie_id, top_n=15)

        # 加权合并
        scores = {}
        for r in cn_recs:
            scores[r['id']] = scores.get(r['id'], 0) + r['similarity'] * 0.35
        for r in ppr_recs:
            scores[r['id']] = scores.get(r['id'], 0) + r['similarity'] * 0.40
        for r in community_recs:
            scores[r['id']] = scores.get(r['id'], 0) + (r['similarity'] / 10.0) * 0.25

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for mid, score in sorted_ids:
            m = self.movie_by_id.get(mid)
            if m:
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': ', '.join(m['genres'][:3]),
                    'similarity': round(float(score), 3),
                    'cover': m['cover'],
                    'poster_local_path': m['poster_local_path'],
                    'method': '图算法融合(公共邻居+PPR+社区)'
                })
        return results
