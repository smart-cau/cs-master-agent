"""
Supabase utility functions for handling resume files and metadata.
Provides helper functions for Storage operations and database queries.
"""
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
