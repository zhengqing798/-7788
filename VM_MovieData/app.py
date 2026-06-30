"""
电影推荐系统 v3.0
- 用户注册/登录
- 用户评分 & 评论
- 管理员后台
- Spark MLlib ALS 协同过滤
- TF-IDF 内容推荐
- 数据可视化大屏
"""
import json
import os
import re
import pickle
import numpy as np
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

# 自定义模块
from database import init_db, get_db
from database import (
    create_user, authenticate_user, get_user_by_id, get_all_users,
    update_user_role, toggle_user_active, get_user_stats,
    add_or_update_rating, get_user_rating, get_movie_avg_rating,
    add_review, get_movie_reviews, get_user_reviews, get_user_ratings,
    delete_review, delete_user_review, delete_user_rating, get_system_stats,
    toggle_like, toggle_favorite, is_liked, is_favorited,
    get_user_likes, get_user_favorites,
    add_browsing_history, get_browsing_history, delete_browsing_history
)
from visualization.data_analyzer import MovieAnalyzer

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# 初始化数据库
init_db()

# ============================================================
# 全局初始化
# ============================================================
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("电影推荐系统 v3.0 启动中...")
print("=" * 60)

# 数据分析器
print("[1/3] 初始化数据分析器...")
analyzer = MovieAnalyzer()

# ALS协同过滤
print("[2/3] 初始化ALS协同过滤...")
from recommendation.collaborative import SparkALSRecommender
als_rec = SparkALSRecommender()
try:
    als_rec.load_or_train(num_users=300, rank=20)
    als_info = als_rec.get_model_info()
    print(f"  ALS状态: {als_info['status']}, 算法: {als_info.get('algorithm', 'N/A')}")
except Exception as e:
    print(f"  ALS异常: {e}")
    als_info = {'status': '初始化失败'}

# TF-IDF内容推荐
print("[3/3] 初始化内容推荐引擎...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

content_similarity = None
id_to_idx = {}
idx_to_id = {}

def build_content_engine():
    global content_similarity, id_to_idx, idx_to_id
    cache_file = os.path.join(CACHE_DIR, 'content_sim.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['similarity'], data['id_to_idx'], data['idx_to_id']

    texts = []
    for m in analyzer.movies:
        parts = []
        if m.get('genres'):
            parts.append(' '.join(m['genres']) * 3)
        parts.append(m.get('title', ''))
        texts.append(' '.join(parts))

    vectorizer = TfidfVectorizer(max_features=3000, analyzer='char_wb',
                                  ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    id2idx = {m['id']: i for i, m in enumerate(analyzer.movies)}
    idx2id = {i: m['id'] for i, m in enumerate(analyzer.movies)}

    with open(cache_file, 'wb') as f:
        pickle.dump({'similarity': sim, 'id_to_idx': id2idx, 'idx_to_id': idx2id}, f)
    return sim, id2idx, idx2id

content_similarity, id_to_idx, idx_to_id = build_content_engine()
print(f"  内容引擎就绪, 矩阵: {content_similarity.shape}")

# 完整电影数据
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movies_data_temp(1).json')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    full_movie_data = json.load(f)
full_movie_by_id = {}
for m in full_movie_data:
    if m.get('id') and m.get('title'):
        full_movie_by_id[str(m['id'])] = m

# 图算法推荐引擎
print("[4/5] 初始化图算法推荐引擎...")
from recommendation.graph_engine import GraphRecommender
graph_rec = GraphRecommender()
graph_rec.build_graph()
print(f"  图引擎就绪: {graph_rec.graph.number_of_nodes()} 节点, {graph_rec.graph.number_of_edges()} 边")

# 用户偏好综合推荐引擎
print("[5/5] 初始化用户偏好推荐引擎...")
from recommendation.user_preference import UserPreferenceRecommender
pref_rec = UserPreferenceRecommender(als_rec, graph_rec, content_similarity, id_to_idx, idx_to_id, full_movie_data)
print("  用户偏好推荐引擎就绪 (Spark MLlib + 图算法 + TF-IDF)")

print("\n系统就绪！ http://127.0.0.1:5000")
print("=" * 60)


# ============================================================
# 辅助函数
# ============================================================
def get_full_movie(movie_id):
    return full_movie_by_id.get(str(movie_id))

def movie_to_card(m):
    # 提取国家/地区列表
    country_str = m.get('country', '')
    countries = [c.strip() for c in country_str.replace('/', ',').split(',') if c.strip()]

    # 提取上映年份
    release_date_str = m.get('release_date', '')
    year_match = re.search(r'(\d{4})', release_date_str) if release_date_str else None
    year = int(year_match.group(1)) if year_match else None

    return {
        'id': str(m.get('id', '')),
        'title': m.get('title', ''),
        'cover': m.get('cover', ''),
        'rating': float(m.get('rating', 0)) if m.get('rating') else 0,
        'genre': m.get('genre', ''),
        'genres': [g.strip() for g in m.get('genre', '').replace('/', ',').split(',') if g.strip()],
        'director': m.get('director', ''),
        'cast': m.get('cast', ''),
        'release_date': release_date_str,
        'poster_local_path': m.get('poster_local_path', ''),
        'runtime': m.get('runtime', ''),
        'summary': m.get('summary', ''),
        'countries': countries,
        'year': year,
    }

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """管理员验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login_page'))
        if session.get('role') != 'admin':
            flash('需要管理员权限', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' in session:
        return get_user_by_id(session['user_id'])
    return None


# ============================================================
# 认证路由
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'欢迎回来，{username}！', 'success')
            next_page = request.args.get('next', '/')
            return redirect(next_page)
        else:
            flash('用户名或密码错误', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        email = request.form.get('email', '').strip()

        if not username or not password:
            flash('用户名和密码不能为空', 'danger')
        elif len(username) < 2 or len(username) > 20:
            flash('用户名长度需在2-20个字符之间', 'danger')
        elif len(password) < 4:
            flash('密码长度至少4个字符', 'danger')
        elif password != confirm:
            flash('两次输入密码不一致', 'danger')
        else:
            success, msg = create_user(username, password, email)
            if success:
                flash(msg + '，请登录', 'success')
                return redirect(url_for('login_page'))
            else:
                flash(msg, 'danger')

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('已退出登录', 'info')
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    user = get_current_user()
    stats = get_user_stats(user['id'])
    ratings = get_user_ratings(user['id'], limit=30)
    reviews = get_user_reviews(user['id'], limit=30)
    likes = get_user_likes(user['id'], limit=50)
    favorites = get_user_favorites(user['id'], limit=50)
    history = get_browsing_history(user['id'], limit=50)

    # 补充电影标题
    for r in ratings:
        m = get_full_movie(r['movie_id'])
        r['movie_title'] = m['title'] if m else '未知电影'
    for r in reviews:
        m = get_full_movie(r['movie_id'])
        r['movie_title'] = m['title'] if m else '未知电影'
    for item in likes:
        m = get_full_movie(item['movie_id'])
        item['movie_title'] = m['title'] if m else '未知电影'
    for item in favorites:
        m = get_full_movie(item['movie_id'])
        item['movie_title'] = m['title'] if m else '未知电影'
    for item in history:
        m = get_full_movie(item['movie_id'])
        item['movie_title'] = m['title'] if m else '未知电影'

    return render_template('profile.html', user=user, stats=stats,
                         ratings=ratings, reviews=reviews,
                         likes=likes, favorites=favorites, history=history)


# ============================================================
# 管理员路由
# ============================================================
@app.route('/admin')
@admin_required
def admin_panel():
    stats = get_system_stats()
    users = get_all_users()
    return render_template('admin.html', stats=stats, users=users)


@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@admin_required
def admin_update_role(user_id):
    role = request.form.get('role', 'user')
    update_user_role(user_id, role)
    flash('角色已更新', 'success')
    return redirect(url_for('admin_panel'))


@app.route('/admin/user/<int:user_id>/toggle', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    toggle_user_active(user_id)
    flash('用户状态已切换', 'success')
    return redirect(url_for('admin_panel'))


@app.route('/api/review/<int:review_id>/delete', methods=['POST'])
@login_required
def api_delete_own_review(review_id):
    """用户删除自己的评论"""
    success = delete_user_review(review_id, session['user_id'])
    if success:
        return jsonify({'success': True, 'message': '评论已删除'})
    else:
        return jsonify({'error': '无权删除此评论'}), 403


@app.route('/admin/review/<int:review_id>/delete', methods=['POST'])
@admin_required
def admin_delete_review(review_id):
    delete_review(review_id)
    flash('评论已删除', 'success')
    return redirect(url_for('admin_panel'))


# ============================================================
# 喜欢 / 收藏 API
# ============================================================
@app.route('/api/like', methods=['POST'])
@login_required
def api_toggle_like():
    data = request.get_json()
    movie_id = data.get('movie_id', '')
    if not movie_id:
        return jsonify({'error': '缺少movie_id'}), 400
    is_liked_now, like_count = toggle_like(session['user_id'], movie_id)
    return jsonify({'success': True, 'is_liked': is_liked_now, 'count': like_count})


@app.route('/api/favorite', methods=['POST'])
@login_required
def api_toggle_favorite():
    data = request.get_json()
    movie_id = data.get('movie_id', '')
    if not movie_id:
        return jsonify({'error': '缺少movie_id'}), 400
    is_fav_now, fav_count = toggle_favorite(session['user_id'], movie_id)
    return jsonify({'success': True, 'is_favorited': is_fav_now, 'count': fav_count})


# ============================================================
# 浏览历史 API
# ============================================================
@app.route('/api/history/delete/<int:history_id>', methods=['POST'])
@login_required
def api_delete_history(history_id):
    success = delete_browsing_history(history_id, session['user_id'])
    if success:
        return jsonify({'success': True, 'message': '已删除'})
    return jsonify({'error': '无权删除此记录'}), 403


# ============================================================
# 评分删除 API
# ============================================================
@app.route('/api/rating/delete', methods=['POST'])
@login_required
def api_delete_rating():
    data = request.get_json()
    movie_id = data.get('movie_id', '')
    if not movie_id:
        return jsonify({'error': '缺少movie_id'}), 400
    success = delete_user_rating(session['user_id'], movie_id)
    if success:
        # 返回删除后的平均评分
        new_stats = get_movie_avg_rating(movie_id)
        return jsonify({'success': True, 'message': '评分已删除', 'avg_rating': new_stats['avg'], 'count': new_stats['count']})
    return jsonify({'error': '未找到该评分'}), 404


# ============================================================
# 数据导入 (管理员)
# ============================================================
@app.route('/admin/import', methods=['GET'])
@admin_required
def admin_import_page():
    """数据导入页面"""
    import_page = request.args.get('page', 'import')
    return render_template('admin_import.html', page=import_page)


@app.route('/admin/import/upload', methods=['POST'])
@admin_required
def admin_import_upload():
    """上传并导入电影数据"""
    if 'datafile' not in request.files:
        flash('请选择文件', 'danger')
        return redirect(url_for('admin_import_page'))

    file = request.files['datafile']
    if file.filename == '':
        flash('请选择文件', 'danger')
        return redirect(url_for('admin_import_page'))

    filename = file.filename.lower()
    try:
        if filename.endswith('.json'):
            # 保存上传的JSON
            upload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movies_data_temp(1).json')
            file.save(upload_path)
            # 验证JSON格式
            with open(upload_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            movie_count = len(data) if isinstance(data, list) else 0

        elif filename.endswith('.csv'):
            # 转换CSV为JSON
            import csv
            upload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movies_data_temp(1).json')
            movies = []
            content = file.read().decode('utf-8-sig')
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                movie = {
                    'id': row.get('id', ''),
                    'title': row.get('title', ''),
                    'cover': row.get('cover', ''),
                    'rating': row.get('rating', ''),
                    'director': row.get('director', ''),
                    'screenwriter': row.get('screenwriter', ''),
                    'cast': row.get('cast', ''),
                    'genre': row.get('genre', ''),
                    'country': row.get('country', ''),
                    'language': row.get('language', ''),
                    'release_date': row.get('release_date', ''),
                    'runtime': row.get('runtime', ''),
                    'imdb': row.get('imdb', ''),
                    'aka': row.get('aka', ''),
                    'summary': row.get('summary', ''),
                    'hot_comments': row.get('hot_comments', ''),
                    'poster_local_path': row.get('poster_local_path', ''),
                }
                movies.append(movie)

            with open(upload_path, 'w', encoding='utf-8') as f:
                json.dump(movies, f, ensure_ascii=False, indent=2)
            movie_count = len(movies)
        else:
            flash('仅支持JSON和CSV格式', 'danger')
            return redirect(url_for('admin_import_page'))

        # 清除缓存
        import shutil
        cache_dir = 'cache'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        flash(f'数据导入成功！共导入 {movie_count} 部电影。缓存已清除，请重启服务器以重建推荐引擎。', 'success')
    except Exception as e:
        flash(f'导入失败: {str(e)}', 'danger')

    return redirect(url_for('admin_import_page'))


@app.route('/admin/import/reload', methods=['POST'])
@admin_required
def admin_reload_data():
    """重新加载数据并重建引擎"""
    try:
        global full_movie_data, full_movie_by_id, content_similarity, id_to_idx, idx_to_id

        # 清除缓存
        import shutil
        cache_dir = 'cache'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        # 重新加载数据
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            full_movie_data = json.load(f)
        full_movie_by_id = {str(m['id']): m for m in full_movie_data if m.get('id') and m.get('title')}

        # 重建内容引擎
        content_similarity, id_to_idx, idx_to_id = build_content_engine()

        # 重建ALS引擎
        global als_rec, als_info
        als_rec = SparkALSRecommender()
        als_rec.load_or_train(num_users=300, rank=20)
        als_info = als_rec.get_model_info()

        # 重建分析器
        global analyzer
        analyzer = MovieAnalyzer()

        flash(f'数据重载成功！当前共 {len(full_movie_data)} 部电影，引擎已重建。', 'success')
    except Exception as e:
        flash(f'重载失败: {str(e)}', 'danger')

    return redirect(url_for('admin_import_page'))


# ============================================================
# 评分 & 评论 API
# ============================================================
@app.route('/api/rate', methods=['POST'])
@login_required
def api_rate():
    data = request.get_json()
    movie_id = data.get('movie_id')
    score = data.get('score')

    if not movie_id or not score:
        return jsonify({'error': '参数不完整'}), 400

    try:
        score = int(score)
        if score < 1 or score > 10:
            raise ValueError
    except ValueError:
        return jsonify({'error': '评分需在1-10之间'}), 400

    add_or_update_rating(session['user_id'], movie_id, score)
    stats = get_movie_avg_rating(movie_id)
    return jsonify({'success': True, 'avg_rating': stats['avg'], 'count': stats['count']})


@app.route('/api/review', methods=['POST'])
@login_required
def api_review():
    data = request.get_json()
    movie_id = data.get('movie_id')
    content = data.get('content', '').strip()

    if not movie_id or not content:
        return jsonify({'error': '参数不完整'}), 400
    if len(content) < 2:
        return jsonify({'error': '评论至少2个字符'}), 400
    if len(content) > 500:
        return jsonify({'error': '评论最多500个字符'}), 400

    add_review(session['user_id'], movie_id, content)
    reviews = get_movie_reviews(movie_id)
    return jsonify({'success': True, 'reviews': reviews})


@app.route('/api/movie/<movie_id>/ratings')
def api_movie_ratings(movie_id):
    """获取电影的评分和评论"""
    stats = get_movie_avg_rating(movie_id)
    reviews = get_movie_reviews(movie_id)
    user_rating = None
    if 'user_id' in session:
        user_rating = get_user_rating(session['user_id'], movie_id)
    return jsonify({
        'avg_rating': stats['avg'],
        'rating_count': stats['count'],
        'reviews': reviews,
        'user_rating': user_rating
    })


# ============================================================
# 首页
# ============================================================
@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    genre = request.args.get('genre', '')
    country = request.args.get('country', '')
    period = request.args.get('period', '')
    sort = request.args.get('sort', 'rating')
    per_page = 24

    # 时间段定义
    PERIODS = [
        ('2020s', 2020, 2029),
        ('2010s', 2010, 2019),
        ('2000s', 2000, 2009),
        ('1990s', 1990, 1999),
        ('1980s', 1980, 1989),
        ('1970s及更早', 0, 1979),
    ]

    # 全部有效电影
    all_movies = [movie_to_card(m) for m in full_movie_data if m.get('title') and m.get('id')]

    # ---- 分面计数：每个维度的数量基于其他两个维度的当前筛选条件 ----
    def filter_other_dims(movies, skip_g=False, skip_c=False, skip_p=False):
        """对 movies 应用当前筛选条件，但跳过指定维度"""
        result = movies
        if genre and not skip_g:
            result = [m for m in result if genre in m['genres']]
        if country and not skip_c:
            result = [m for m in result if country in m['countries']]
        if period and not skip_p:
            p_info = next((pi for pi in PERIODS if pi[0] == period), None)
            if p_info:
                _, ps, pe = p_info
                result = [m for m in result if m['year'] and ps <= m['year'] <= pe]
        return result

    # 全维度排序列表（基于全部电影，保证选项始终可见）
    all_genres = sorted(set(g for m in all_movies for g in m['genres']),
                       key=lambda g: sum(1 for m in all_movies if g in m['genres']), reverse=True)
    all_countries = sorted(set(c for m in all_movies for c in m['countries']),
                          key=lambda c: sum(1 for m in all_movies if c in m['countries']), reverse=True)

    # 类型计数：基于 地区 + 年代 当前筛选
    genre_base = filter_other_dims(all_movies, skip_g=True)
    genre_counts = {g: sum(1 for m in genre_base if g in m['genres']) for g in all_genres}

    # 地区计数：基于 类型 + 年代 当前筛选
    country_base = filter_other_dims(all_movies, skip_c=True)
    country_counts = {c: sum(1 for m in country_base if c in m['countries']) for c in all_countries}

    # 年代计数：基于 类型 + 地区 当前筛选
    period_base = filter_other_dims(all_movies, skip_p=True)
    period_counts = {}
    for p_name, ps, pe in PERIODS:
        period_counts[p_name] = sum(1 for m in period_base if m['year'] and ps <= m['year'] <= pe)

    # 各维度"全部"标签对应的数量（基于其他维度的筛选结果）
    total_for_genre_all = len(genre_base)
    total_for_country_all = len(country_base)
    total_for_period_all = len(period_base)

    # ---- 叠加筛选：应用所有维度 ----
    movies = filter_other_dims(all_movies)

    if sort == 'rating':
        movies.sort(key=lambda x: x['rating'], reverse=True)
    elif sort == 'date':
        movies.sort(key=lambda x: x['release_date'], reverse=True)
    elif sort == 'runtime':
        movies.sort(key=lambda x: int(re.search(r'\d+', x['runtime']).group()) if re.search(r'\d+', x.get('runtime', '')) else 0, reverse=True)

    total = len(movies)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_movies = movies[start:start + per_page]

    stats = analyzer.summary_stats()
    user = get_current_user()

    return render_template('index.html', movies=page_movies, genres=all_genres,
                         genre_counts=genre_counts,
                         all_countries=all_countries,
                         country_counts=country_counts,
                         periods=[p[0] for p in PERIODS],
                         period_counts=period_counts,
                         total_for_genre_all=total_for_genre_all,
                         total_for_country_all=total_for_country_all,
                         total_for_period_all=total_for_period_all,
                         current_genre=genre, current_country=country, current_period=period,
                         current_sort=sort,
                         page=page, total_pages=total_pages, total=total,
                         statistics=stats, user=user)


# ============================================================
# 电影详情页
# ============================================================
@app.route('/movie/<movie_id>')
def movie_detail(movie_id):
    movie = get_full_movie(movie_id)
    if not movie:
        return render_template('404.html'), 404

    card = movie_to_card(movie)
    user = get_current_user()

    # 用户评分
    user_rating = None
    liked = False
    favorited = False
    if user:
        user_rating = get_user_rating(user['id'], movie_id)
        liked = is_liked(user['id'], movie_id)
        favorited = is_favorited(user['id'], movie_id)
        # 记录浏览历史
        add_browsing_history(user['id'], movie_id)

    # 影片评分统计
    rating_stats = get_movie_avg_rating(movie_id)

    # 用户评论
    user_reviews = get_movie_reviews(movie_id)

    # 内容推荐
    recs = []
    idx = id_to_idx.get(str(movie_id))
    if idx is not None:
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        for i, score in sim_scores[1:7]:
            mid = idx_to_id.get(i)
            m = get_full_movie(mid)
            if m:
                m_card = movie_to_card(m)
                m_card['similarity'] = round(float(score), 3)
                recs.append(m_card)

    return render_template('detail.html', movie=card, movie_raw=movie,
                         recommendations=recs, user=user,
                         user_rating=user_rating, rating_stats=rating_stats,
                         user_reviews=user_reviews,
                         is_liked=liked, is_favorited=favorited)


# ============================================================
# 推荐中心
# ============================================================
@app.route('/recommendations')
@login_required
def user_recommendations():
    """用户偏好推荐页面"""
    user = get_current_user()
    results, profile = pref_rec.recommend(user['id'], top_n=24)
    analysis = pref_rec.analyze_profile(user['id'])

    return render_template('preferences.html',
                         user=user,
                         recommendations=results,
                         profile=analysis['profile'],
                         top_genres=analysis['top_genres'],
                         top_decades=analysis['top_decades'],
                         top_countries=analysis['top_countries'])


@app.route('/recommend/<movie_id>')
def recommend_page(movie_id):
    movie = get_full_movie(movie_id)
    if not movie:
        return render_template('404.html'), 404

    user = get_current_user()
    return render_template('recommend.html', movie=movie_to_card(movie), user=user)


@app.route('/api/recommend/all/<movie_id>')
def api_recommend_all(movie_id):
    result = {'movie_id': str(movie_id)}

    # 内容推荐
    idx = id_to_idx.get(str(movie_id))
    if idx is not None:
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        result['content_based'] = []
        for i, score in sim_scores[1:13]:
            mid = idx_to_id.get(i)
            m = get_full_movie(mid)
            if m:
                result['content_based'].append({
                    **movie_to_card(m),
                    'similarity': round(float(score), 3),
                    'method': 'TF-IDF内容推荐'
                })

    # ALS协同过滤
    try:
        result['als_collaborative'] = als_rec.recommend_for_movie(movie_id, top_n=12)
    except Exception as e:
        result['als_collaborative'] = []
        result['als_error'] = str(e)

    return jsonify(result)


# ============================================================
# 可视化大屏
# ============================================================
@app.route('/dashboard')
def dashboard():
    stats = analyzer.summary_stats()
    user = get_current_user()
    return render_template('dashboard.html', statistics=stats, user=user)


@app.route('/api/dashboard/full_report')
def api_full_report():
    return jsonify(analyzer.full_report())


@app.route('/api/dashboard/<report_name>')
def api_report(report_name):
    method_map = {
        'rating_distribution': analyzer.rating_distribution,
        'genre_distribution': analyzer.genre_distribution,
        'country_distribution': analyzer.country_distribution,
        'year_trend': analyzer.year_trend,
        'runtime_distribution': analyzer.runtime_distribution,
        'rating_vs_runtime': analyzer.rating_vs_runtime,
        'language_distribution': analyzer.language_distribution,
        'director_ranking': analyzer.director_ranking,
        'actor_ranking': analyzer.actor_ranking,
        'summary_stats': analyzer.summary_stats,
    }
    if report_name in method_map:
        return jsonify(method_map[report_name]())
    return jsonify({'error': '未知报告类型'}), 404


# ============================================================
# 排行榜
# ============================================================
@app.route('/top')
def top_charts():
    movies = [movie_to_card(m) for m in full_movie_data if m.get('title') and m.get('id')]

    # 提取所有类型，按电影数量排序
    all_genres = sorted(set(g for m in movies for g in m['genres']),
                       key=lambda g: sum(1 for m in movies if g in m['genres']), reverse=True)

    # 构建类型计数
    genre_counts = {g: sum(1 for m in movies if g in m['genres']) for g in all_genres}

    # 当前选中的类型（默认为第一个，即电影最多的类型）
    current_genre = request.args.get('genre', all_genres[0] if all_genres else '')

    # 只获取当前选中类型的 Top 20
    genre_movies = [m for m in movies if current_genre in m['genres']]
    genre_movies.sort(key=lambda x: x['rating'], reverse=True)
    ranking = {
        'genre': current_genre,
        'count': len(genre_movies),
        'movies': genre_movies[:20],
    }

    user = get_current_user()
    return render_template('top.html',
                         all_genres=all_genres,
                         genre_counts=genre_counts,
                         current_genre=current_genre,
                         ranking=ranking,
                         user=user)


# ============================================================
# 人物页面（导演 / 演员作品）
# ============================================================
@app.route('/person')
def person_page():
    name = request.args.get('name', '').strip()
    ptype = request.args.get('type', 'actor')  # 'director' 或 'actor'

    if not name:
        return render_template('person.html', name='', ptype=ptype, movies=[], total=0, user=get_current_user())

    movies = []
    for m in full_movie_data:
        if not m.get('title') or not m.get('id'):
            continue
        field = m.get('director', '') if ptype == 'director' else m.get('cast', '')
        # 按 / 分割后匹配（支持单人匹配，也支持包含关系）
        persons = [p.strip() for p in field.replace('/', ',').split(',') if p.strip()]
        if name in persons:
            movies.append(movie_to_card(m))

    # 按评分排序
    movies.sort(key=lambda x: x['rating'], reverse=True)

    user = get_current_user()
    return render_template('person.html', name=name, ptype=ptype,
                         movies=movies, total=len(movies), user=user)


# ============================================================
# 搜索
# ============================================================
@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = []
    if query:
        ql = query.lower()
        for m in full_movie_data:
            if not m.get('title'):
                continue
            score = 0
            if ql in m['title'].lower():
                score += 10
            if ql in m.get('director', '').lower():
                score += 5
            if ql in m.get('cast', '').lower():
                score += 3
            if ql in m.get('genre', '').lower():
                score += 2
            if score > 0:
                results.append((movie_to_card(m), score))
        results.sort(key=lambda x: x[1], reverse=True)
        results = [m for m, _ in results[:50]]

    user = get_current_user()
    return render_template('search.html', query=query, results=results, user=user)


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': [], 'count': 0})
    ql = query.lower()
    results = []
    for m in full_movie_data:
        if not m.get('title'):
            continue
        score = 0
        if ql in m['title'].lower():
            score += 10
        if ql in m.get('director', '').lower():
            score += 5
        if ql in m.get('cast', '').lower():
            score += 3
        if ql in m.get('genre', '').lower():
            score += 2
        if score > 0:
            results.append((m, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return jsonify({
        'query': query,
        'count': len(results),
        'results': [movie_to_card(m) for m, _ in results[:20]]
    })


# ============================================================
# API
# ============================================================
@app.route('/api/als/info')
def api_als_info():
    return jsonify(als_info)


# ============================================================
# 错误处理
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


