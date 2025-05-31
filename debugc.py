"""
Celery debugging script to identify configuration issues.
"""

import sys
import os
import traceback

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test required imports"""
    print("=== Testing Imports ===")
    
    try:
        import celery
        print(f"✅ Celery version: {celery.__version__}")
    except ImportError as e:
        print(f"❌ Celery import failed: {e}")
        return False
    
    try:
        import redis
        print(f"✅ Redis-py version: {redis.__version__}")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    try:
        import psycopg2
        print(f"✅ Psycopg2 version: {psycopg2.__version__}")
    except ImportError as e:
        print(f"⚠️  Psycopg2 import failed: {e}")
        print("   This is needed if using PostgreSQL as result backend")
    
    try:
        import sqlalchemy
        print(f"✅ SQLAlchemy version: {sqlalchemy.__version__}")
    except ImportError as e:
        print(f"⚠️  SQLAlchemy import failed: {e}")
        print("   This might be needed for PostgreSQL result backend")
    
    return True

def test_redis_connection():
    """Test Redis connection"""
    print("\n=== Testing Redis Connection ===")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("\n=== Testing PostgreSQL Connection ===")
    
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            dbname="yourdb", 
            user="youruser",
            password="yourpassword"
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except ImportError as e:
        print(f"❌ Psycopg2 not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        from core.config import get_config
        config = get_config()
        print(f"✅ Config loaded successfully")
        print(f"   Broker URL: {config.celery.broker_url}")
        print(f"   Result Backend: {config.celery.result_backend}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_celery_app():
    """Test Celery app creation"""
    print("\n=== Testing Celery App Creation ===")
    
    try:
        from core.celery_app import celery_app
        print(f"✅ Celery app created successfully")
        print(f"   App name: {celery_app.main}")
        print(f"   Broker URL: {celery_app.conf.broker_url}")
        print(f"   Result Backend: {celery_app.conf.result_backend}")
        return True
    except Exception as e:
        print(f"❌ Celery app creation failed: {e}")
        traceback.print_exc()
        return False

def test_task_creation():
    """Test creating a simple task"""
    print("\n=== Testing Task Creation ===")
    
    try:
        from core.celery_app import test_task
        print("✅ Test task imported successfully")
        
        # Try to create task instance (don't execute)
        task_instance = test_task.s()
        print("✅ Task instance created successfully")
        return True
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n=== Suggested Fixes ===")
    
    print("1. Install missing packages:")
    print("   pip install psycopg2-binary sqlalchemy")
    print("   pip install 'celery[redis,postgresql]'")
    
    print("\n2. Alternative - Use Redis for everything:")
    print("   Change result_backend to: redis://localhost:6379/1")
    
    print("\n3. Check if services are running:")
    print("   - Redis: redis-cli ping")
    print("   - PostgreSQL: pg_isready -h localhost -p 5433")
    
    print("\n4. Verify connection strings match your setup")

def main():
    """Run all diagnostic tests"""
    print("🔍 Celery Configuration Diagnostics")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_redis_connection():
        all_passed = False
    
    test_postgres_connection()  # Optional for now
    
    if not test_config_loading():
        all_passed = False
    
    if not test_celery_app():
        all_passed = False
    
    if not test_task_creation():
        all_passed = False
    
    if not all_passed:
        suggest_fixes()
        sys.exit(1)
    else:
        print("\n🎉 All diagnostics passed! Celery should be working.")

if __name__ == "__main__":
    main()