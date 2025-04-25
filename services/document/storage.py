import os
import json
import logging
import shutil
import boto3
import pickle
from typing import Any, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class StorageAdapter:
    """Base class for storage operations, abstracting file I/O."""
    
    def write_file(self, content: Any, path: str) -> bool:
        """Write content to a file path."""
        raise NotImplementedError
        
    def read_file(self, path: str) -> Any:
        """Read content from a file path."""
        raise NotImplementedError
        
    def delete_file(self, path: str) -> bool:
        """Delete a file at the given path."""
        raise NotImplementedError
        
    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        raise NotImplementedError
        
    def create_directory(self, path: str) -> bool:
        """Create a directory at the given path."""
        raise NotImplementedError
        
    def list_files(self, directory: str) -> list:
        """List files in a directory."""
        raise NotImplementedError
        
    def delete_directory(self, path: str) -> bool:
        """Delete a directory and its contents."""
        raise NotImplementedError

class LocalStorageAdapter(StorageAdapter):
    """Local filesystem implementation of StorageAdapter."""
    
    def write_file(self, content: Any, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if isinstance(content, str):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            elif isinstance(content, bytes):
                with open(path, 'wb') as f:
                    f.write(content)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(content, f)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {path}: {str(e)}")
            return False
    
    def read_file(self, path: str) -> Any:
        if not os.path.exists(path):
            return None
        
        try:
            if path.endswith('.pkl'):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            elif any(path.endswith(ext) for ext in ['.jpg', '.png', '.pdf', '.bin']):
                with open(path, 'rb') as f:
                    return f.read()
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return None
    
    def delete_file(self, path: str) -> bool:
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {path}: {str(e)}")
            return False
    
    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)
    
    def create_directory(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return False
    
    def list_files(self, directory: str) -> list:
        try:
            if os.path.exists(directory):
                return os.listdir(directory)
            return []
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {str(e)}")
            return []
    
    def delete_directory(self, path: str) -> bool:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting directory {path}: {str(e)}")
            return False

class S3StorageAdapter(StorageAdapter):
    """S3 implementation of StorageAdapter."""
    
    def __init__(self, bucket_name: str, prefix: str = "", region: str = "us-east-1"):
        self.bucket = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=region)
    
    def _get_full_path(self, path: str) -> str:
        """Convert local path to S3 key."""
        # Remove leading slash if present
        if path.startswith('/'):
            path = path[1:]
        # Add prefix if it exists
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{path}"
        return path
    
    def write_file(self, content: Any, path: str) -> bool:
        s3_key = self._get_full_path(path)
        try:
            if isinstance(content, str):
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content.encode('utf-8')
                )
            elif isinstance(content, bytes):
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content
                )
            else:
                # Serialize with pickle for other types
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=pickle.dumps(content)
                )
            return True
        except Exception as e:
            logger.error(f"Error writing to S3 {s3_key}: {str(e)}")
            return False
    
    def read_file(self, path: str) -> Any:
        s3_key = self._get_full_path(path)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response['Body'].read()
            
            if path.endswith('.pkl'):
                return pickle.loads(content)
            elif any(path.endswith(ext) for ext in ['.jpg', '.png', '.pdf', '.bin']):
                return content
            else:
                return content.decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"Error reading from S3 {s3_key}: {str(e)}")
            return None
    
    def delete_file(self, path: str) -> bool:
        s3_key = self._get_full_path(path)
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from S3 {s3_key}: {str(e)}")
            return False
    
    def file_exists(self, path: str) -> bool:
        s3_key = self._get_full_path(path)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def create_directory(self, path: str) -> bool:
        # S3 doesn't need directory creation, but we add a marker
        s3_key = self._get_full_path(path.rstrip('/') + '/.marker')
        try:
            self.s3_client.put_object(Bucket=self.bucket, Key=s3_key, Body=b'')
            return True
        except Exception as e:
            logger.error(f"Error creating S3 directory {path}: {str(e)}")
            return False
    
    def list_files(self, directory: str) -> list:
        # Ensure directory ends with a slash
        if not directory.endswith('/'):
            directory += '/'
        
        s3_prefix = self._get_full_path(directory)
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=s3_prefix,
                Delimiter='/'
            )
            
            files = []
            
            # Get files (Contents)
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract just the filename from the full path
                    key = obj['Key']
                    file_name = key.split('/')[-1]
                    if file_name and file_name != '.marker':
                        files.append(file_name)
            
            return files
        except Exception as e:
            logger.error(f"Error listing S3 files {s3_prefix}: {str(e)}")
            return []
    
    def delete_directory(self, path: str) -> bool:
        s3_prefix = self._get_full_path(path.rstrip('/') + '/')
        try:
            # S3 doesn't have directories, delete all objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            
            delete_list = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        delete_list.append({'Key': obj['Key']})
            
            if delete_list:
                # Delete in batches of 1000 (S3 limit)
                for i in range(0, len(delete_list), 1000):
                    batch = delete_list[i:i+1000]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket,
                        Delete={'Objects': batch}
                    )
                return True
            return True  # Nothing to delete is still success
        except Exception as e:
            logger.error(f"Error deleting S3 directory {s3_prefix}: {str(e)}")
            return False