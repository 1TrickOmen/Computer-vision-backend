"""
Database migration script for Computer Vision Backend Service
"""

import asyncio
from pathlib import Path
from app.core.config import settings
from app.core.database import init_db


async def create_tables():
    """Create all database tables"""
    print("🔄 Creating database tables...")
    
    # Create database directory if it doesn't exist
    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    await init_db()
    print("✅ Database tables created successfully!")


async def main():
    """Main migration function"""
    print("🚀 Starting database migration...")
    
    # Create tables
    await create_tables()
    
    print("🎉 Database migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
