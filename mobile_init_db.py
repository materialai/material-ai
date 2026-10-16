"""
移动端知识库初始化脚本
将现有数据库复制到移动端可打包路径，供Flet APK使用
"""
import os
import sys

def ensure_mobile_db():
    """确保存在可用于打包的SQLite DB"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "pcm_knowledge.db")
    
    if not os.path.exists(db_path):
        print("📦 知识库不存在，开始初始化...")
        # 导入初始数据
        sys.path.insert(0, current_dir)
        try:
            from import_pcm_sqlite import init_default_db
            init_default_db()
        except Exception as e:
            print(f"默认知识库初始化失败: {e}")
        
        try:
            from import_bulk_knowledge import init_bulk_db
            init_bulk_db()
        except Exception as e:
            print(f"全品类知识导入失败: {e}")
    
    if os.path.exists(db_path):
        size_kb = os.path.getsize(db_path) / 1024
        print(f"✅ 知识库就绪: {db_path} ({size_kb:.1f} KB)")
        return True
    else:
        print("❌ 知识库不存在")
        return False


if __name__ == "__main__":
    ensure_mobile_db()
