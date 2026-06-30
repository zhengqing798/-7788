"""
Spark MLlib ALS 协同过滤推荐模块
使用交替最小二乘法(ALS)进行电影推荐
"""
import json
import os
import pickle
import numpy as np
from collections import defaultdict


class SparkALSRecommender:
    """
    基于Spark MLlib ALS的协同过滤推荐器
    包含Spark实现和纯Python备用实现
    """

    def __init__(self, data_path=None, cache_dir='cache'):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'movies_data_temp(1).json')
        self.data_path = data_path
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cache_dir) if not os.path.isabs(cache_dir) else cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.movies = []
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_factors = None  # ALS user latent factors
        self.item_factors = None  # ALS item latent factors
        self.model_trained = False

        self._load_data()
        self._check_spark()

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for i, m in enumerate(raw):
            if m.get('title') and m.get('id'):
                try:
                    rating = float(m.get('rating', 0))
                except (ValueError, TypeError):
                    rating = 0.0
                self.movies.append({
                    'id': str(m['id']),
                    'title': m['title'].strip(),
                    'rating': rating,
                    'genre': m.get('genre', ''),
                    'cover': m.get('cover', ''),
                    'poster_local_path': m.get('poster_local_path', ''),
                })
                self.movie_id_to_idx[str(m['id'])] = i
                self.idx_to_movie_id[i] = str(m['id'])

    def _check_spark(self):
        """检查Spark是否可用"""
        try:
            import pyspark
            self.spark_available = True
            print(f"PySpark {pyspark.__version__} 可用")
        except ImportError:
            self.spark_available = False
            print("PySpark 不可用，使用NumPy备用实现")

    # ================================================================
    # 模拟用户评分数据
    # ================================================================
    def simulate_ratings(self, num_users=500, ratings_per_user=50):
        """
        模拟用户评分矩阵
        基于电影真实评分+类型偏好+噪声生成用户评分
        """
        print(f"模拟 {num_users} 个用户评分数据...")
        np.random.seed(42)

        # 为每个用户随机分配类型偏好
        all_genres = set()
        for m in self.movies:
            genres = [g.strip() for g in m['genre'].replace('/', ',').split(',') if g.strip()]
            all_genres.update(genres)
        all_genres = list(all_genres)

        user_genre_prefs = np.random.dirichlet(np.ones(len(all_genres)), size=num_users)

        ratings = []
        for user_id in range(num_users):
            # 用户偏好的类型
            top_genre_idx = np.argsort(user_genre_prefs[user_id])[-3:]  # top 3 genres

            # 候选电影评分
            movie_scores = []
            for i, m in enumerate(self.movies):
                m_genres = [g.strip() for g in m['genre'].replace('/', ',').split(',') if g.strip()]
                genre_match = 0
                for g in m_genres:
                    if g in [all_genres[idx] for idx in top_genre_idx]:
                        genre_match += 1

                # 基础分 = 电影真实评分 + 类型匹配加成 + 噪声
                base_score = m['rating'] * 0.6 + genre_match * 1.5 + np.random.normal(0, 1.5)
                base_score = max(1, min(10, base_score))
                movie_scores.append((i, base_score))

            # 每个用户评价 top-N 部匹配度最高的电影
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            selected = np.random.choice(
                len(movie_scores),
                size=min(ratings_per_user, len(movie_scores)),
                p=self._softmax([s for _, s in movie_scores]) if len(movie_scores) <= 2000
                else None
            )
            # If not using softmax sampling (too many movies), pick top matches
            if len(movie_scores) > 2000:
                # Pick from top 200 with probability, plus random sampling
                top_n = min(200, len(movie_scores))
                selected = list(range(top_n))
                if ratings_per_user > top_n:
                    extra = np.random.choice(
                        range(top_n, len(movie_scores)),
                        size=min(ratings_per_user - top_n, len(movie_scores) - top_n),
                        replace=False
                    )
                    selected.extend(extra)
                selected = selected[:ratings_per_user]

            for idx in selected:
                if isinstance(idx, (int, np.integer)):
                    i = idx
                actual_rating = max(1, min(10, int(round(movie_scores[i][1]))))
                ratings.append((user_id, i, float(actual_rating)))

        print(f"生成 {len(ratings)} 条评分记录")
        return ratings

    def _softmax(self, x):
        e_x = np.exp(np.array(x) - np.max(x))
        return e_x / e_x.sum()

    # ================================================================
    # 方法1: Spark MLlib ALS 实现
    # ================================================================
    def train_spark_als(self, ratings, rank=20, max_iter=15, reg_param=0.1):
        """使用Spark MLlib ALS训练模型"""
        if not self.spark_available:
            print("Spark不可用，回退到NumPy实现")
            return self.train_numpy_als(ratings, rank=rank, max_iter=max_iter, reg_param=reg_param)

        try:
            from pyspark.sql import SparkSession
            from pyspark.ml.recommendation import ALS
            from pyspark.ml.evaluation import RegressionEvaluator

            spark = SparkSession.builder \
                .appName("MovieALS") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.shuffle.partitions", "4") \
                .master("local[*]") \
                .getOrCreate()

            # 创建DataFrame
            ratings_df = spark.createDataFrame(
                ratings, schema=["userId", "movieId", "rating"]
            )

            # 划分训练集/测试集
            train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)

            # 构建ALS模型
            als = ALS(
                maxIter=max_iter,
                rank=rank,
                regParam=reg_param,
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy="drop",
                nonnegative=True
            )

            print(f"训练Spark ALS: rank={rank}, iter={max_iter}, reg={reg_param}")
            model = als.fit(train)

            # 评估
            predictions = model.transform(test)
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction"
            )
            rmse = evaluator.evaluate(predictions)
            print(f"ALS训练完成，RMSE={rmse:.4f}")

            # 提取因子矩阵
            item_factors = model.itemFactors().toPandas()
            user_factors = model.userFactors().toPandas()

            # 转换为numpy数组
            n_items = len(self.movies)
            n_users = user_factors['id'].max() + 1
            k = rank

            self.item_factors = np.zeros((n_items, k))
            for _, row in item_factors.iterrows():
                idx = int(row['id'])
                if idx < n_items:
                    self.item_factors[idx] = np.array(row['features'])

            self.user_factors = np.zeros((n_users, k))
            for _, row in user_factors.iterrows():
                uid = int(row['id'])
                if uid < n_users:
                    self.user_factors[uid] = np.array(row['features'])

            # 缓存
            self.model_trained = True
            with open(os.path.join(self.cache_dir, 'als_factors.pkl'), 'wb') as f:
                pickle.dump({
                    'item_factors': self.item_factors,
                    'user_factors': self.user_factors,
                    'rank': rank, 'rmse': rmse
                }, f)

            spark.stop()
            return rmse

        except Exception as e:
            print(f"Spark ALS出错: {e}")
            print("回退到NumPy实现")
            return self.train_numpy_als(ratings, rank=rank, max_iter=max_iter, reg_param=reg_param)

    # ================================================================
    # 方法2: NumPy ALS 备用实现 (Spark不可用时)
    # ================================================================
    def train_numpy_als(self, ratings, rank=20, max_iter=15, reg_param=0.1):
        """纯NumPy实现的ALS算法（Spark MLlib算法的等效实现）"""
        print(f"训练NumPy ALS: rank={rank}, iter={max_iter}")

        # 构建评分矩阵
        n_users = max(r[0] for r in ratings) + 1
        n_items = len(self.movies)

        # 稀疏矩阵
        rating_matrix = np.zeros((n_users, n_items))
        rating_mask = np.zeros((n_users, n_items), dtype=bool)
        for u, i, r in ratings:
            if i < n_items:
                rating_matrix[u, i] = r
                rating_mask[u, i] = True

        # 初始化因子矩阵
        np.random.seed(42)
        U = np.random.normal(0, 0.1, (n_users, rank))
        V = np.random.normal(0, 0.1, (n_items, rank))

        # ALS迭代
        for iteration in range(max_iter):
            # 固定V, 更新U
            for u in range(n_users):
                rated_items = np.where(rating_mask[u])[0]
                if len(rated_items) == 0:
                    continue
                V_u = V[rated_items]
                R_u = rating_matrix[u, rated_items]
                A = V_u.T @ V_u + reg_param * np.eye(rank) * len(rated_items)
                b = V_u.T @ R_u
                try:
                    U[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    U[u] = np.linalg.lstsq(A, b, rcond=None)[0]

            # 固定U, 更新V
            for i in range(n_items):
                rated_users = np.where(rating_mask[:, i])[0]
                if len(rated_users) == 0:
                    continue
                U_i = U[rated_users]
                R_i = rating_matrix[rated_users, i]
                A = U_i.T @ U_i + reg_param * np.eye(rank) * len(rated_users)
                b = U_i.T @ R_i
                try:
                    V[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    V[i] = np.linalg.lstsq(A, b, rcond=None)[0]

            # 计算RMSE
            if iteration % 5 == 0 or iteration == max_iter - 1:
                pred = U @ V.T
                errors = (pred[rating_mask] - rating_matrix[rating_mask]) ** 2
                rmse = np.sqrt(errors.mean())
                print(f"  Iter {iteration+1}/{max_iter}, RMSE={rmse:.4f}")

        self.user_factors = U
        self.item_factors = V
        self.model_trained = True

        # 缓存
        with open(os.path.join(self.cache_dir, 'als_factors.pkl'), 'wb') as f:
            pickle.dump({
                'item_factors': self.item_factors,
                'user_factors': self.user_factors,
                'rank': rank,
            }, f)

        return rmse

    # ================================================================
    # 推荐方法
    # ================================================================
    def load_or_train(self, num_users=300, rank=20):
        """加载缓存模型或重新训练"""
        cache_file = os.path.join(self.cache_dir, 'als_factors.pkl')
        if os.path.exists(cache_file):
            print("加载缓存的ALS模型...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.item_factors = data['item_factors']
                self.user_factors = data['user_factors']
                self.model_trained = True
            return True

        print("训练新的ALS模型...")
        ratings = self.simulate_ratings(num_users=num_users)
        if self.spark_available:
            self.train_spark_als(ratings, rank=rank)
        else:
            self.train_numpy_als(ratings, rank=rank)
        return True

    def recommend_for_movie(self, movie_id, top_n=12):
        """基于物品隐因子的相似电影推荐（Item-based CF）"""
        if not self.model_trained:
            self.load_or_train()

        idx = self.movie_id_to_idx.get(str(movie_id))
        if idx is None or self.item_factors is None:
            return []

        target_vec = self.item_factors[idx]

        # 计算余弦相似度
        norm = np.linalg.norm(self.item_factors, axis=1)
        target_norm = np.linalg.norm(target_vec)

        if target_norm == 0:
            return []

        similarities = self.item_factors @ target_vec / (norm * target_norm + 1e-8)

        # 排序取top
        sim_idxs = np.argsort(similarities)[::-1]
        results = []
        for i in sim_idxs:
            if i == idx:
                continue
            mid = self.idx_to_movie_id.get(i)
            if mid and similarities[i] > 0:
                m = self.movies[i]
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': m['genre'],
                    'similarity': round(float(similarities[i]), 3),
                    'method': 'ALS协同过滤',
                    'cover': m.get('cover', ''),
                    'poster_local_path': m.get('poster_local_path', ''),
                })
            if len(results) >= top_n:
                break
        return results

    def recommend_for_user(self, user_id, top_n=12):
        """为用户生成个性化推荐"""
        if not self.model_trained or self.user_factors is None:
            return []

        user_vec = self.user_factors[user_id]
        scores = self.item_factors @ user_vec
        top_idxs = np.argsort(scores)[::-1][:top_n]

        results = []
        for i in top_idxs:
            mid = self.idx_to_movie_id.get(i)
            if mid:
                m = self.movies[i]
                results.append({
                    'id': mid,
                    'title': m['title'],
                    'rating': m['rating'],
                    'genre': m['genre'],
                    'score': round(float(scores[i]), 3),
                    'method': 'ALS个性化推荐'
                })
        return results

    def get_model_info(self):
        """获取模型信息"""
        if not self.model_trained:
            return {'status': '未训练'}
        return {
            'status': '已训练',
            'algorithm': 'Spark MLlib ALS' if self.spark_available else 'NumPy ALS (等效实现)',
            'user_factors_shape': list(self.user_factors.shape) if self.user_factors is not None else None,
            'item_factors_shape': list(self.item_factors.shape) if self.item_factors is not None else None,
            'rank': self.item_factors.shape[1] if self.item_factors is not None else None,
        }
