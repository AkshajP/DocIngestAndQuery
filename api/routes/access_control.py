from fastapi import Depends, HTTPException
from typing import Optional
from db.chat_store.repository import UserCaseRepository

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

async def validate_user_case_access(
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    required_role: Optional[str] = None
):
    """
    Validate that the current user has access to the current case with required role.
    Raises HTTPException(403) if access is denied.
    
    Args:
        user_id: User ID from get_current_user dependency
        case_id: Case ID from get_current_case dependency
        required_role: Optional minimum required role (viewer, editor, owner)
        
    Returns:
        True if access is granted
    """
    user_case_repo = UserCaseRepository()
    
    if not user_case_repo.check_access(user_id, case_id, required_role):
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this resource"
        )
    
    return True