"""
Database connection utilities for AgriTwin-GH
Supports SQLite and PostgreSQL (with optional TimescaleDB)
"""

import os
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import yaml
from dotenv import load_dotenv


class DatabaseManager:
    """Manage database connections and sessions"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            config_path: Path to settings.yaml file
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            config_path = base_dir / 'config' / 'settings.yaml'
            
            # Check for local settings override
            local_config_path = base_dir / 'config' / 'settings.local.yaml'
            if local_config_path.exists():
                config_path = local_config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.engine = None
        self.SessionLocal = None
    
    def get_connection_string(self) -> str:
        """
        Build database connection string from configuration
        
        Returns:
            SQLAlchemy connection string
        """
        db_config = self.config.get('database', {})
        db_type = db_config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            db_path = db_config.get('path', 'data/processed/agritwin.db')
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return f'sqlite:///{db_path}'
        
        elif db_type == 'postgresql':
            # Get credentials from environment or config
            user = os.getenv('DB_USER', db_config.get('user', 'postgres'))
            password = os.getenv('DB_PASSWORD', db_config.get('password', ''))
            host = os.getenv('DB_HOST', db_config.get('host', 'localhost'))
            port = os.getenv('DB_PORT', db_config.get('port', 5432))
            dbname = os.getenv('DB_NAME', db_config.get('name', 'agritwin_db'))
            
            return f'postgresql://{user}:{password}@{host}:{port}/{dbname}'
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def create_engine_instance(self, echo: bool = False):
        """
        Create SQLAlchemy engine
        
        Args:
            echo: Whether to log SQL queries
        """
        connection_string = self.get_connection_string()
        
        # Configure engine based on database type
        if 'postgresql' in connection_string:
            self.engine = create_engine(
                connection_string,
                echo=echo,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True  # Verify connections before using
            )
        else:
            self.engine = create_engine(
                connection_string,
                echo=echo,
                connect_args={'check_same_thread': False}  # SQLite specific
            )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        return self.engine
    
    def get_session(self) -> Session:
        """
        Get a new database session
        
        Returns:
            SQLAlchemy Session instance
        """
        if self.SessionLocal is None:
            self.create_engine_instance()
        
        return self.SessionLocal()
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the singleton database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_engine_instance()
    return _db_manager


def get_db_session() -> Session:
    """
    Get a database session (for dependency injection)
    
    Usage:
        session = get_db_session()
        try:
            # Use session
            pass
        finally:
            session.close()
    """
    db_manager = get_db_manager()
    return db_manager.get_session()
