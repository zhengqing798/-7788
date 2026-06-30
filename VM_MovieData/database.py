"""
数据库模块 - SQLite
用户、评分、评论管理
"""
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = 'movie_system.db'


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()

    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            avatar TEXT DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')

    # 电影评分表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id TEXT NOT NULL,
            score INTEGER NOT NULL CHECK(score >= 1 AND score <= 10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, movie_id)
        )
    ''')

    # 电影评论表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # 用户喜欢表 (小红心)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, movie_id)
        )
    ''')

    # 用户收藏表 (小黄星)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, movie_id)
        )
    ''')

    # 浏览历史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS browsing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id TEXT NOT NULL,
            viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # 评论点赞表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            review_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
            UNIQUE(user_id, review_id)
        )
    ''')

    # 评论回复表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # 喜欢的人物表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_liked_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            person_name TEXT NOT NULL,
            person_type TEXT NOT NULL DEFAULT 'actor' CHECK(person_type IN ('director', 'actor')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, person_name)
        )
    ''')

    # 创建默认管理员
    admin_exists = cursor.execute(
        "SELECT id FROM users WHERE username = ?", ('admin',)
    ).fetchone()

    if not admin_exists:
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
            ('admin', generate_password_hash('admin123'), 'admin@movie.com', 'admin')
        )
        print("默认管理员已创建: admin / admin123")

    conn.commit()
    conn.close()
    print("数据库初始化完成")


# ============================================================
# 用户操作
# ============================================================
def create_user(username, password, email=''):
    """创建新用户"""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), email)
        )
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    finally:
        conn.close()


def authenticate_user(username, password):
    """验证用户登录"""
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
    ).fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None


def get_user_by_id(user_id):
    """根据ID获取用户"""
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_users():
    """获取所有用户（管理员用）"""
    conn = get_db()
    users = conn.execute(
        "SELECT id, username, email, role, created_at, is_active FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(u) for u in users]


def update_user_role(user_id, role):
    """更新用户角色"""
    conn = get_db()
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    conn.close()


def toggle_user_active(user_id):
    """切换用户激活状态"""
    conn = get_db()
    conn.execute(
        "UPDATE users SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()


def get_user_stats(user_id):
    """获取用户统计"""
    conn = get_db()
    rating_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM ratings WHERE user_id = ?", (user_id,)
    ).fetchone()['cnt']
    review_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM reviews WHERE user_id = ?", (user_id,)
    ).fetchone()['cnt']
    avg_rating = conn.execute(
        "SELECT AVG(score) as avg FROM ratings WHERE user_id = ?", (user_id,)
    ).fetchone()['avg']
    like_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM user_likes WHERE user_id = ?", (user_id,)
    ).fetchone()['cnt']
    fav_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM user_favorites WHERE user_id = ?", (user_id,)
    ).fetchone()['cnt']
    conn.close()
    return {
        'rating_count': rating_count,
        'review_count': review_count,
        'avg_rating': round(avg_rating, 1) if avg_rating else 0,
        'like_count': like_count,
        'fav_count': fav_count,
    }


# ============================================================
# 评分操作
# ============================================================
def add_or_update_rating(user_id, movie_id, score):
    """添加或更新评分"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM ratings WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE ratings SET score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (score, existing['id'])
        )
    else:
        conn.execute(
            "INSERT INTO ratings (user_id, movie_id, score) VALUES (?, ?, ?)",
            (user_id, str(movie_id), score)
        )
    conn.commit()
    conn.close()
    return True


def get_user_rating(user_id, movie_id):
    """获取用户对某电影的评分"""
    conn = get_db()
    rating = conn.execute(
        "SELECT score FROM ratings WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    conn.close()
    return rating['score'] if rating else None


def get_movie_avg_rating(movie_id):
    """获取电影平均评分"""
    conn = get_db()
    result = conn.execute(
        "SELECT AVG(score) as avg, COUNT(*) as cnt FROM ratings WHERE movie_id = ?",
        (str(movie_id),)
    ).fetchone()
    conn.close()
    return {
        'avg': round(result['avg'], 1) if result['avg'] else 0,
        'count': result['cnt']
    }


def get_user_ratings(user_id, limit=50):
    """获取用户的所有评分"""
    conn = get_db()
    ratings = conn.execute(
        "SELECT movie_id, score, updated_at FROM ratings WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in ratings]


# ============================================================
# 评论操作
# ============================================================
def add_review(user_id, movie_id, content):
    """添加评论"""
    conn = get_db()
    conn.execute(
        "INSERT INTO reviews (user_id, movie_id, content) VALUES (?, ?, ?)",
        (user_id, str(movie_id), content)
    )
    conn.commit()
    conn.close()
    return True


def get_movie_reviews(movie_id, limit=20):
    """获取电影的用户评论"""
    conn = get_db()
    reviews = conn.execute(
        """SELECT r.id, r.content, r.created_at, u.username, u.avatar
           FROM reviews r JOIN users u ON r.user_id = u.id
           WHERE r.movie_id = ?
           ORDER BY r.created_at DESC LIMIT ?""",
        (str(movie_id), limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reviews]


def get_user_reviews(user_id, limit=50):
    """获取用户的所有评论"""
    conn = get_db()
    reviews = conn.execute(
        """SELECT r.id, r.movie_id, r.content, r.created_at
           FROM reviews r WHERE r.user_id = ?
           ORDER BY r.created_at DESC LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reviews]


def delete_review(review_id):
    """删除评论（管理员用）"""
    conn = get_db()
    conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()


def delete_user_review(review_id, user_id):
    """删除自己的评论"""
    conn = get_db()
    review = conn.execute(
        "SELECT id FROM reviews WHERE id = ? AND user_id = ?",
        (review_id, user_id)
    ).fetchone()
    if review:
        conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


# ============================================================
# 喜欢 / 收藏 操作
# ============================================================
def toggle_like(user_id, movie_id):
    """切换喜欢状态，返回 (is_liked, like_count)"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM user_likes WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    if existing:
        conn.execute("DELETE FROM user_likes WHERE id = ?", (existing['id'],))
        is_liked = False
    else:
        conn.execute(
            "INSERT INTO user_likes (user_id, movie_id) VALUES (?, ?)",
            (user_id, str(movie_id))
        )
        is_liked = True
    like_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM user_likes WHERE movie_id = ?", (str(movie_id),)
    ).fetchone()['cnt']
    conn.commit()
    conn.close()
    return is_liked, like_count


def toggle_favorite(user_id, movie_id):
    """切换收藏状态，返回 (is_favorited, fav_count)"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM user_favorites WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    if existing:
        conn.execute("DELETE FROM user_favorites WHERE id = ?", (existing['id'],))
        is_favorited = False
    else:
        conn.execute(
            "INSERT INTO user_favorites (user_id, movie_id) VALUES (?, ?)",
            (user_id, str(movie_id))
        )
        is_favorited = True
    fav_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM user_favorites WHERE movie_id = ?", (str(movie_id),)
    ).fetchone()['cnt']
    conn.commit()
    conn.close()
    return is_favorited, fav_count


def is_liked(user_id, movie_id):
    """检查用户是否已喜欢"""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM user_likes WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    conn.close()
    return row is not None


def is_favorited(user_id, movie_id):
    """检查用户是否已收藏"""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM user_favorites WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    conn.close()
    return row is not None


def get_user_likes(user_id, limit=50):
    """获取用户喜欢列表"""
    conn = get_db()
    rows = conn.execute(
        "SELECT movie_id, created_at FROM user_likes WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_favorites(user_id, limit=50):
    """获取用户收藏列表"""
    conn = get_db()
    rows = conn.execute(
        "SELECT movie_id, created_at FROM user_favorites WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# 浏览历史 操作
# ============================================================
def add_browsing_history(user_id, movie_id):
    """添加或更新浏览记录（已有则更新时间戳）"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM browsing_history WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE browsing_history SET viewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (existing['id'],)
        )
    else:
        conn.execute(
            "INSERT INTO browsing_history (user_id, movie_id) VALUES (?, ?)",
            (user_id, str(movie_id))
        )
    conn.commit()
    conn.close()


def get_browsing_history(user_id, limit=50):
    """获取用户浏览历史"""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, movie_id, viewed_at FROM browsing_history WHERE user_id = ? ORDER BY viewed_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_browsing_history(history_id, user_id):
    """删除单条浏览记录"""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM browsing_history WHERE id = ? AND user_id = ?",
        (history_id, user_id)
    ).fetchone()
    if row:
        conn.execute("DELETE FROM browsing_history WHERE id = ?", (history_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


# ============================================================
# 评分删除 操作
# ============================================================
def delete_user_rating(user_id, movie_id):
    """删除用户的评分"""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM ratings WHERE user_id = ? AND movie_id = ?",
        (user_id, str(movie_id))
    ).fetchone()
    if row:
        conn.execute("DELETE FROM ratings WHERE id = ?", (row['id'],))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


# ============================================================
# 统计
# ============================================================
def get_system_stats():
    """系统统计"""
    conn = get_db()
    stats = {
        'total_users': conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()['cnt'],
        'total_ratings': conn.execute("SELECT COUNT(*) as cnt FROM ratings").fetchone()['cnt'],
        'total_reviews': conn.execute("SELECT COUNT(*) as cnt FROM reviews").fetchone()['cnt'],
        'active_users': conn.execute("SELECT COUNT(*) as cnt FROM users WHERE is_active = 1").fetchone()['cnt'],
        'avg_user_rating': 0,
        'recent_users': [],
        'recent_reviews': [],
    }

    avg_r = conn.execute("SELECT AVG(score) as avg FROM ratings").fetchone()['avg']
    if avg_r:
        stats['avg_user_rating'] = round(avg_r, 1)

    stats['recent_users'] = [
        dict(u) for u in conn.execute(
            "SELECT id, username, email, created_at FROM users ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
    ]

    stats['recent_reviews'] = [
        dict(r) for r in conn.execute(
            """SELECT r.id, r.content, r.created_at, r.movie_id, u.username
               FROM reviews r JOIN users u ON r.user_id = u.id
               ORDER BY r.created_at DESC LIMIT 10"""
        ).fetchall()
    ]

    conn.close()
    return stats


# ============================================================
# 喜欢的人物操作
# ============================================================
def toggle_like_person(user_id, person_name, person_type):
    """切换喜欢/取消喜欢人物"""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM user_liked_persons WHERE user_id = ? AND person_name = ?",
        (user_id, person_name)
    ).fetchone()
    if existing:
        conn.execute("DELETE FROM user_liked_persons WHERE id = ?", (existing['id'],))
        conn.commit()
        conn.close()
        return False  # 已取消
    else:
        conn.execute(
            "INSERT INTO user_liked_persons (user_id, person_name, person_type) VALUES (?, ?, ?)",
            (user_id, person_name, person_type)
        )
        conn.commit()
        conn.close()
        return True  # 已喜欢


def is_person_liked(user_id, person_name):
    """检查是否已喜欢某人物"""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM user_liked_persons WHERE user_id = ? AND person_name = ?",
        (user_id, person_name)
    ).fetchone()
    conn.close()
    return row is not None


def get_liked_persons(user_id, person_type=None, limit=50):
    """获取用户喜欢的人物列表"""
    conn = get_db()
    if person_type:
        rows = conn.execute(
            "SELECT person_name, person_type, created_at FROM user_liked_persons WHERE user_id = ? AND person_type = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, person_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT person_name, person_type, created_at FROM user_liked_persons WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# 评论点赞
# ============================================================
def toggle_review_like(user_id, review_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM review_likes WHERE user_id = ? AND review_id = ?", (user_id, review_id)
    ).fetchone()
    if row:
        conn.execute("DELETE FROM review_likes WHERE id = ?", (row['id'],))
        conn.commit(); conn.close(); return False
    else:
        conn.execute("INSERT INTO review_likes (user_id, review_id) VALUES (?,?)", (user_id, review_id))
        conn.commit(); conn.close(); return True


def get_review_likes(review_id):
    conn = get_db()
    cnt = conn.execute("SELECT COUNT(*) as c FROM review_likes WHERE review_id = ?", (review_id,)).fetchone()['c']
    conn.close()
    return cnt


def is_review_liked(user_id, review_id):
    conn = get_db()
    row = conn.execute("SELECT id FROM review_likes WHERE user_id = ? AND review_id = ?", (user_id, review_id)).fetchone()
    conn.close()
    return row is not None


# ============================================================
# 评论回复
# ============================================================
def add_review_reply(review_id, user_id, content):
    conn = get_db()
    conn.execute("INSERT INTO review_replies (review_id, user_id, content) VALUES (?,?,?)",
                 (review_id, user_id, content))
    conn.commit()
    conn.close()
    return True


def get_review_replies(review_id):
    conn = get_db()
    rows = conn.execute(
        """SELECT r.id, r.content, r.created_at, u.username
           FROM review_replies r JOIN users u ON r.user_id = u.id
           WHERE r.review_id = ? ORDER BY r.created_at ASC""",
        (review_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_review_reply(reply_id, user_id):
    conn = get_db()
    conn.execute("DELETE FROM review_replies WHERE id = ? AND user_id = ?", (reply_id, user_id))
    conn.commit()
    conn.close()


def update_user_profile(user_id, username=None, email=None, avatar=None):
    """更新用户资料"""
    conn = get_db()
    if username:
        conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
    if email is not None:
        conn.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
    if avatar is not None:
        conn.execute("UPDATE users SET avatar = ? WHERE id = ?", (avatar, user_id))
    conn.commit()
    conn.close()
    return True


# 启动时初始化
if __name__ == '__main__':
    init_db()
else:
    if not os.path.exists(DB_PATH):
        init_db()
    else:
        init_db()
