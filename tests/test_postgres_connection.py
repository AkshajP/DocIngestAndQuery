# db/chat_store/test_connection.py

import psycopg2
import psycopg2.extras
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_test")

def test_connection(host="localhost", port=5432, dbname="document_rag_db", 
                   user="postgres", password="postgres"):
    """
    Test connection to PostgreSQL database.
    
    Args:
        host: Database host
        port: Database port
        dbname: Database name
        user: Database user
        password: Database password
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Attempting to connect to PostgreSQL database {dbname} on {host}:{port}")
        
        # Connect to the database
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        logger.info("Connection established successfully!")
        
        # Get database info
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"PostgreSQL version: {version}")
            
            # Check if our tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            tables = cursor.fetchall()
            if tables:
                logger.info("Found the following tables:")
                for table in tables:
                    logger.info(f"  - {table[0]}")
                
                # Check for our specific tables
                expected_tables = ['chats', 'chat_documents', 'messages', 'feedback', 'user_case_access']
                found_tables = [t[0] for t in tables]
                
                for table in expected_tables:
                    if table in found_tables:
                        logger.info(f"✅ Table '{table}' exists")
                    else:
                        logger.warning(f"❌ Table '{table}' is missing")
            else:
                logger.warning("No tables found in the database!")
        
        # Test inserting and retrieving a record (in a transaction that we'll rollback)
        logger.info("Testing database operations...")
        
        with conn.cursor() as cursor:
            # Start a transaction that we'll roll back
            try:
                # Create a test chat
                test_chat_id = f"test_chat_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                cursor.execute(
                    """
                    INSERT INTO chats (chat_id, title, user_id, case_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING chat_id
                    """,
                    (test_chat_id, "Test Chat", "test_user", "test_case", datetime.now(), datetime.now())
                )
                
                returned_id = cursor.fetchone()[0]
                
                if returned_id == test_chat_id:
                    logger.info("✅ Test record inserted successfully!")
                else:
                    logger.warning("❌ Test record insertion returned unexpected result")
                
                # Now try to retrieve it
                cursor.execute(
                    "SELECT * FROM chats WHERE chat_id = %s",
                    (test_chat_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    logger.info("✅ Test record retrieved successfully!")
                else:
                    logger.warning("❌ Failed to retrieve test record")
                
            except Exception as e:
                logger.error(f"Error during operation test: {str(e)}")
            finally:
                # Always roll back the test transaction
                logger.info("Rolling back test transaction...")
                conn.rollback()
        
        # Close the connection
        conn.close()
        logger.info("Database connection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return False

def main():
    """Main function to run the test with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PostgreSQL database connection')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5433, help='Database port')
    parser.add_argument('--dbname', default='yourdb', help='Database name')
    parser.add_argument('--user', default='youruser', help='Database user')
    parser.add_argument('--password', default='yourpassword', help='Database password')
    
    args = parser.parse_args()
    
    success = test_connection(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()