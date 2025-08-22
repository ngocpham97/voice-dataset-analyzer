#!/usr/bin/env python3
"""
MinIO Uploader for Voice Dataset

This module provides functionality to upload audio files and JSON data to MinIO storage.
Supports batch upload, progress tracking, and various file formats.

Usage:
    python minio_client.py --folder ./dataset --bucket voice-dataset
    python minio_client.py --file dataset.json --bucket results
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import mimetypes
from datetime import datetime
import hashlib
from urllib.parse import urlparse

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    print("Warning: minio package not installed. Install with: pip install minio")


def _sanitize_minio_endpoint(endpoint: Optional[str]) -> Tuple[str, bool]:
    """Sanitize endpoint by stripping scheme and any path. Infer HTTPS from scheme.

    Returns (hostport, secure)
    """
    if not endpoint:
        return 'localhost:9000', False

    ep = endpoint.strip()
    secure = False
    hostport = ep

    if '://' in ep:
        parsed = urlparse(ep)
        hostport = parsed.netloc or parsed.path
        secure = parsed.scheme.lower() == 'https'
    else:
        hostport = ep

    # Remove trailing slashes and any accidental path portions
    hostport = hostport.rstrip('/')
    if '/' in hostport:
        hostport = hostport.split('/')[0]

    return hostport, secure


class MinIOClient:
    def __init__(self, 
                 endpoint: str = None,
                 access_key: str = None,
                 secret_key: str = None,
                 secure: bool = False,
                 region: str = "us-east-1"):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint (host:port or full URL)
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            secure: Use HTTPS if True (overrides scheme inference)
            region: AWS region (optional)
        """
        if not MINIO_AVAILABLE:
            raise ImportError("minio package not installed. Install with: pip install minio")
        
        # Get credentials from environment if not provided
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY')
        endpoint_input = endpoint or os.getenv('MINIO_ENDPOINT')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO credentials not provided. Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY environment variables.")
        
        # Sanitize endpoint and infer secure from scheme if provided
        hostport, inferred_secure = _sanitize_minio_endpoint(endpoint_input)
        self.endpoint = hostport
        self.secure = bool(secure) or inferred_secure
        self.region = region

        print(f"Endpoint: {self.endpoint} (secure={self.secure})")
        
        # Initialize MinIO client
        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
            region=self.region
        )
        
        # Note: defer connection validation to first S3 call to avoid
        # permission/region signature issues on ListBuckets
        print("Client initialized. Connection will be validated on first operation.")
    
    def create_bucket(self, bucket_name: str, region: str = None) -> bool:
        """
        Create a new bucket if it doesn't exist.
        
        Args:
            bucket_name: Name of the bucket to create
            region: Region for the bucket (falls back to client region if not provided)
            
        Returns:
            True if bucket created or exists, False otherwise
        """
        try:
            effective_region = region or self.region
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, region=effective_region)
                print(f"✓ Created bucket: {bucket_name}")
            else:
                print(f"✓ Bucket already exists: {bucket_name}")
            return True
        except S3Error as e:
            print(f"❌ Failed to create bucket {bucket_name}: {e}")
            return False
    
    def upload_file(self, 
                   file_path: str, 
                   bucket_name: str, 
                   object_name: str = None,
                   metadata: Dict = None) -> bool:
        """
        Upload a single file to MinIO.
        
        Args:
            file_path: Local path to the file
            bucket_name: Target bucket name
            object_name: Object name in MinIO (default: filename)
            metadata: Additional metadata to attach
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            # Use filename if object_name not specified
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            # Determine content type
            content_type = self._get_content_type(file_path)
            
            # Prepare metadata
            file_metadata = {
                'original_filename': os.path.basename(file_path),
                'upload_timestamp': datetime.now().isoformat(),
                'file_size': str(os.path.getsize(file_path)),
                'content_type': content_type
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # Upload file
            self.client.fput_object(
                bucket_name, 
                object_name, 
                file_path,
                content_type=content_type,
                metadata=file_metadata
            )
            
            print(f"✓ Uploaded: {file_path} -> {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            print(f"❌ Failed to upload {file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error uploading {file_path}: {e}")
            return False
    
    def upload_folder(self, 
                     folder_path: str, 
                     bucket_name: str, 
                     prefix: str = "",
                     include_patterns: List[str] = None,
                     exclude_patterns: List[str] = None,
                     recursive: bool = True) -> Dict:
        """
        Upload entire folder to MinIO.
        
        Args:
            folder_path: Local folder path
            bucket_name: Target bucket name
            prefix: Prefix for object names in MinIO
            include_patterns: File patterns to include (e.g., ['*.wav', '*.json'])
            exclude_patterns: File patterns to exclude
            recursive: Upload subdirectories recursively
            
        Returns:
            Dictionary with upload statistics
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return {}
        
        # Ensure bucket exists
        if not self.create_bucket(bucket_name):
            return {}
        
        # Collect files to upload
        files_to_upload = self._collect_files(
            folder_path, include_patterns, exclude_patterns, recursive
        )
        
        if not files_to_upload:
            print(f"No files found to upload in {folder_path}")
            return {}
        
        print(f"Found {len(files_to_upload)} files to upload")
        
        # Upload files
        successful_uploads = 0
        failed_uploads = 0
        total_size = 0
        
        for file_path in files_to_upload:
            # Calculate relative path for object name
            rel_path = os.path.relpath(file_path, folder_path)
            object_name = os.path.join(prefix, rel_path).replace('\\', '/')
            
            # Upload file
            if self.upload_file(file_path, bucket_name, object_name):
                successful_uploads += 1
                total_size += os.path.getsize(file_path)
            else:
                failed_uploads += 1
        
        # Compile results
        results = {
            'bucket_name': bucket_name,
            'folder_path': folder_path,
            'total_files': len(files_to_upload),
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'upload_timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n📊 UPLOAD SUMMARY")
        print(f"Bucket: {bucket_name}")
        print(f"Folder: {folder_path}")
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful_uploads']}")
        print(f"Failed: {results['failed_uploads']}")
        print(f"Total size: {results['total_size_mb']} MB")
        
        return results
    
    def upload_json_data(self, 
                         data: Dict, 
                         bucket_name: str, 
                         object_name: str,
                         metadata: Dict = None) -> bool:
        """
        Upload JSON data directly to MinIO.
        
        Args:
            data: Dictionary data to upload
            bucket_name: Target bucket name
            object_name: Object name in MinIO
            metadata: Additional metadata
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, indent=2, ensure_ascii=False)
            json_bytes = json_data.encode('utf-8')
            
            # Prepare metadata
            file_metadata = {
                'content_type': 'application/json',
                'upload_timestamp': datetime.now().isoformat(),
                'data_size': str(len(json_bytes)),
                'data_type': 'json'
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # Upload JSON data
            self.client.put_object(
                bucket_name,
                object_name,
                json_bytes,
                length=len(json_bytes),
                content_type='application/json',
                metadata=file_metadata
            )
            
            print(f"✓ Uploaded JSON data: {bucket_name}/{object_name}")
            return True
            
        except S3Error as e:
            print(f"❌ Failed to upload JSON data: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error uploading JSON: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = "", recursive: bool = True) -> List[Dict]:
        """
        List objects in a bucket.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Prefix to filter objects
            recursive: List recursively
            
        Returns:
            List of object information dictionaries
        """
        try:
            objects = []
            for obj in self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive):
                objects.append({
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                })
            return objects
        except S3Error as e:
            print(f"❌ Failed to list objects: {e}")
            return []
    
    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            bucket_name: Source bucket name
            object_name: Object name in MinIO
            file_path: Local path to save the file
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Download file
            self.client.fget_object(bucket_name, object_name, file_path)
            print(f"✓ Downloaded: {bucket_name}/{object_name} -> {file_path}")
            return True
            
        except S3Error as e:
            print(f"❌ Failed to download {object_name}: {e}")
            return False
    
    def download_folder(self, bucket_name: str, prefix: str, local_folder: str, recursive: bool = True) -> int:
        """
        Download all files from a MinIO bucket/prefix to a local folder.

        Args:
            bucket_name: Name of the bucket
            prefix: Prefix/folder in bucket to download
            local_folder: Local folder to save files
            recursive: Download recursively

        Returns:
            Number of files downloaded
        """
        os.makedirs(local_folder, exist_ok=True)
        objects = self.list_objects(bucket_name, prefix=prefix, recursive=recursive)
        count = 0
        for obj in objects:
            local_path = os.path.join(local_folder, os.path.relpath(obj['name'], prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            success = self.download_file(bucket_name, obj['name'], local_path)
            if success:
                count += 1
        print(f"\n✓ Downloaded {count} files from '{bucket_name}/{prefix}' to '{local_folder}'")
        return count

    def _collect_files(self, 
                      folder_path: str, 
                      include_patterns: List[str] = None,
                      exclude_patterns: List[str] = None,
                      recursive: bool = True) -> List[str]:
        """Collect files to upload based on patterns."""
        import glob
        
        files = []
        
        # Default include patterns
        if include_patterns is None:
            include_patterns = ['*']
        
        # Collect files
        for pattern in include_patterns:
            if recursive:
                pattern_path = os.path.join(folder_path, '**', pattern)
                files.extend(glob.glob(pattern_path, recursive=True))
            else:
                pattern_path = os.path.join(folder_path, pattern)
                files.extend(glob.glob(pattern_path))
        
        # Filter out directories
        files = [f for f in files if os.path.isfile(f)]
        
        # Apply exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                files = [f for f in files if not glob.fnmatch.fnmatch(os.path.basename(f), pattern)]
        
        return sorted(files)
    
    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension."""
        content_type, _ = mimetypes.guess_type(file_path)
        
        # Default content types for common audio formats
        if content_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            content_types = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.flac': 'audio/flac',
                '.m4a': 'audio/mp4',
                '.ogg': 'audio/ogg',
                '.json': 'application/json',
                '.txt': 'text/plain',
                '.csv': 'text/csv'
            }
            content_type = content_types.get(ext, 'application/octet-stream')
        
        return content_type


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Upload files to MinIO')
    parser.add_argument('--endpoint', help='MinIO endpoint')
    parser.add_argument('--access-key', help='MinIO access key')
    parser.add_argument('--secret-key', help='MinIO secret key')
    parser.add_argument('--secure', action='store_true', help='Use HTTPS')
    parser.add_argument('--bucket', required=True, help='Target bucket name')
    
    # Upload options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', help='Folder to upload')
    group.add_argument('--file', help='Single file to upload')
    group.add_argument('--json', help='JSON file to upload as data')
    group.add_argument('--download-folder', help='Local folder to save downloaded files from MinIO')

    parser.add_argument('--prefix', default='', help='Prefix for object names')
    parser.add_argument('--include', nargs='*', help='File patterns to include')
    parser.add_argument('--exclude', nargs='*', help='File patterns to exclude')
    parser.add_argument('--recursive', action='store_true', default=True, help='Upload recursively')
    parser.add_argument('--download-prefix', default='', help='Prefix/folder in bucket to download')
    
    args = parser.parse_args()
    
    try:
        # Initialize uploader
        client = MinIOClient(
            endpoint=args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
            secure=args.secure
        )
        
        # Perform upload based on arguments
        if args.folder:
            # Upload folder
            results = client.upload_folder(
                folder_path=args.folder,
                bucket_name=args.bucket,
                prefix=args.prefix,
                include_patterns=args.include,
                exclude_patterns=args.exclude,
                recursive=args.recursive
            )
            
            # Save upload results
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"upload_results_{timestamp}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Upload results saved to: {results_file}")
        
        elif args.file:
            # Upload single file
            object_name = args.prefix + os.path.basename(args.file) if args.prefix else None
            success = client.upload_file(args.file, args.bucket, object_name)
            if success:
                print("✓ File upload completed successfully")
            else:
                print("❌ File upload failed")
        
        elif args.json:
            # Upload JSON file as data
            with open(args.json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            object_name = args.prefix + "dataset.json" if args.prefix else "dataset.json"
            success = client.upload_json_data(data, args.bucket, object_name)
            if success:
                print("✓ JSON data upload completed successfully")
            else:
                print("❌ JSON data upload failed")
        
        elif args.download_folder:
            # Download folder
            success = client.download_folder(args.bucket, args.download_prefix, args.download_folder, args.recursive)
            if success > 0:
                print("✓ Folder download completed successfully")
            else:
                print("❌ Folder download failed or no files found")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())