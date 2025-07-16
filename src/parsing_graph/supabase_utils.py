"""
Supabase utility functions for handling resume files and metadata.
Provides helper functions for Storage operations and database queries.
"""

import base64
from typing import Optional, Dict, Any, List
from constants.supabase import supabase


class SupabaseError(Exception):
    """Supabase 관련 에러를 위한 커스텀 예외 클래스"""
    pass


class FileAccessError(SupabaseError):
    """파일 접근 권한 관련 에러"""
    pass


class FileNotFoundError(SupabaseError):
    """파일을 찾을 수 없는 에러"""
    pass


class FileDownloadError(SupabaseError):
    """파일 다운로드 실패 에러"""
    pass


def get_latest_resume_file(user_id: str) -> Optional[Dict[str, Any]]:
    """
    사용자의 최신 이력서 파일 정보를 조회합니다.
    
    Args:
        user_id: 사용자 UUID
        
    Returns:
        최신 이력서 레코드 딕셔너리 또는 None
        
    Raises:
        SupabaseError: Supabase 연결 또는 쿼리 실패 시
    """
    try:
        response = supabase.table("resumes").select("*").eq("user_id", user_id).order("uploaded_at", desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error fetching latest resume: {e}")
        raise SupabaseError(f"최신 이력서 조회 실패: {str(e)}")


def get_resume_file_by_path(user_id: str, file_path: str) -> Optional[Dict[str, Any]]:
    """
    특정 파일 경로의 이력서 정보를 조회합니다.
    
    Args:
        user_id: 사용자 UUID
        file_path: Storage 내 파일 경로
        
    Returns:
        이력서 레코드 딕셔너리 또는 None
        
    Raises:
        FileAccessError: 파일에 접근 권한이 없는 경우
        SupabaseError: Supabase 연결 또는 쿼리 실패 시
    """
    try:
        response = supabase.table("resumes").select("*").eq("user_id", user_id).eq("file_path", file_path).single().execute()
        
        return response.data if response.data else None
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "no rows" in error_msg:
            return None
        print(f"Error fetching resume by path: {e}")
        raise SupabaseError(f"이력서 파일 정보 조회 실패: {str(e)}")


def download_resume_file(file_path: str) -> Optional[bytes]:
    """
    Storage에서 이력서 파일을 다운로드합니다.
    
    Args:
        file_path: Storage 내 파일 경로
        
    Returns:
        파일의 바이트 데이터 또는 None
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        FileDownloadError: 파일 다운로드 실패 시
    """
    try:
        file_data = supabase.storage.from_("resumes").download(file_path)
        return file_data
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        print(f"Error downloading file: {e}")
        raise FileDownloadError(f"파일 다운로드 실패: {str(e)}")


def download_resume_as_base64(file_path: str) -> Optional[str]:
    """
    Storage에서 이력서 파일을 다운로드하고 base64로 인코딩합니다.
    
    Args:
        file_path: Storage 내 파일 경로
        
    Returns:
        Base64 인코딩된 문자열 또는 None
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        FileDownloadError: 파일 다운로드 또는 인코딩 실패 시
    """
    try:
        file_data = download_resume_file(file_path)
        if file_data:
            return base64.b64encode(file_data).decode("utf-8")
        return None
    except (FileNotFoundError, FileDownloadError):
        raise  # 이미 구체적인 에러이므로 다시 raise
    except Exception as e:
        print(f"Error encoding file to base64: {e}")
        raise FileDownloadError(f"파일 Base64 인코딩 실패: {str(e)}")


def validate_user_access_to_file(user_id: str, file_path: str) -> bool:
    """
    사용자가 특정 파일에 접근 권한이 있는지 확인합니다.
    
    Args:
        user_id: 사용자 UUID
        file_path: Storage 내 파일 경로
        
    Returns:
        접근 권한 여부
        
    Raises:
        SupabaseError: Supabase 연결 실패 시
    """
    try:
        resume_record = get_resume_file_by_path(user_id, file_path)
        return resume_record is not None
    except SupabaseError:
        raise  # 이미 구체적인 에러이므로 다시 raise
    except Exception as e:
        print(f"Error validating user access: {e}")
        raise SupabaseError(f"파일 접근 권한 확인 실패: {str(e)}") 