"""
Cache management for data layer.
"""

from typing import Any, Dict, Optional
import time
import os
import json
import hashlib

from ..core.base import BaseStorage
from ..core.config import DEFAULT_CONFIG


class CacheManager(BaseStorage):
    """Simple file-based cache manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager."""
        self.config = DEFAULT_CONFIG
        if config:
            self.config = self.config.from_dict(config)
        
        self.cache_dir = self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def _is_expired(self, cache_path: str) -> bool:
        """Check if cache file is expired."""
        if not os.path.exists(cache_path):
            return True
        
        file_time = os.path.getmtime(cache_path)
        current_time = time.time()
        return (current_time - file_time) > self.config.cache_ttl
    
    def store(self, key: str, value: Any) -> bool:
        """Store value in cache."""
        try:
            cache_path = self._get_cache_path(key)
            cache_data = {
                'key': key,
                'value': value,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"Error storing in cache: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        try:
            cache_path = self._get_cache_path(key)
            
            if self._is_expired(cache_path):
                return None
            
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            return cache_data['value']
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return True
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_path = self._get_cache_path(key)
        return os.path.exists(cache_path) and not self._is_expired(cache_path)
    
    def list_keys(self, pattern: str = "*") -> list:
        """List all cache keys."""
        try:
            keys = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    if not self._is_expired(cache_path):
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                            keys.append(cache_data['key'])
            return keys
        except Exception as e:
            print(f"Error listing cache keys: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
