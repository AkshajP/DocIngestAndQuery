from fastapi import Depends, HTTPException
from typing import Optional
from db.chat_store.repository import UserCaseRepository
from core.service_manager import get_initialized_service_manager
import logging
logger = logging.getLogger(__name__)
# Dependency functions (to be replaced with actual auth middleware)
async def get_current_user():
    # This would normally verify the token and return the user_id
    return "user_test"

async def get_current_case():
    # This would normally get the current case from the request
    return "default"

# Dependency for admin authentication
async def get_admin_user():
    # This would normally verify the token and ensure admin privileges
    return "admin_user"

def get_user_case_repository() -> UserCaseRepository:
    """Get shared UserCaseRepository from service manager"""
    service_manager = get_initialized_service_manager()
    return service_manager.user_case_repository

async def validate_user_case_access(
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    user_case_repo: UserCaseRepository = Depends(get_user_case_repository),
    required_role: Optional[str] = None
) -> bool:
    """
    Validate that the user has access to the specified case, optionally with a required role.
    
    Args:
        user_id: User ID from header
        case_id: Case ID from header  
        user_case_repo: Shared UserCaseRepository instance
        required_role: Optional minimum required role ("owner", "editor", "viewer")
                      If None, any access is sufficient
        
    Returns:
        True if access is valid
        
    Raises:
        HTTPException: If access is denied or insufficient role
    """
    try:
        # Check if user has access to the case (with optional role requirement)
        has_access = user_case_repo.check_access(user_id, case_id, required_role)
        
        if not has_access:
            if required_role:
                logger.warning(f"Access denied: User {user_id} does not have {required_role} access to case {case_id}")
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions: {required_role} role required for case {case_id}"
                )
            else:
                logger.warning(f"Access denied: User {user_id} does not have access to case {case_id}")
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to case {case_id}"
                )
        
        if required_role:
            logger.debug(f"Access granted: User {user_id} has {required_role} access to case {case_id}")
        else:
            logger.debug(f"Access granted: User {user_id} has access to case {case_id}")
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user case access: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error validating access permissions"
        )