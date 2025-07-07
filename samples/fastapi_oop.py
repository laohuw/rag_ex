from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import uvicorn
from enum import Enum


# ===============================
# DTOs and Models (same as before)
# ===============================

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)


class UpdateUserRequest(BaseModel):
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    status: Optional[UserStatus] = None


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class User:
    def __init__(self, username: str, email: str, full_name: str,
                 age: Optional[int] = None, user_id: Optional[str] = None):
        self.id = user_id or str(uuid.uuid4())
        self.username = username
        self.email = email
        self.full_name = full_name
        self.age = age
        self.status = UserStatus.ACTIVE
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'age': self.age,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


# ===============================
# Repository and Service (simplified)
# ===============================

class IUserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> User: pass

    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[User]: pass

    @abstractmethod
    def find_all(self) -> List[User]: pass

    @abstractmethod
    def delete_by_id(self, user_id: str) -> bool: pass

    @abstractmethod
    def exists_by_username(self, username: str) -> bool: pass

    @abstractmethod
    def exists_by_email(self, email: str) -> bool: pass


class InMemoryUserRepository(IUserRepository):
    def __init__(self):
        self._users: Dict[str, User] = {}

    def save(self, user: User) -> User:
        self._users[user.id] = user
        return user

    def find_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def find_all(self) -> List[User]:
        return list(self._users.values())

    def delete_by_id(self, user_id: str) -> bool:
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False

    def exists_by_username(self, username: str) -> bool:
        return any(user.username == username for user in self._users.values())

    def exists_by_email(self, email: str) -> bool:
        return any(user.email == email for user in self._users.values())


class UserService:
    def __init__(self, user_repository: IUserRepository):
        self._user_repository = user_repository

    def create_user(self, request: CreateUserRequest) -> User:
        if self._user_repository.exists_by_username(request.username):
            raise ValueError(f"Username '{request.username}' already exists")

        if self._user_repository.exists_by_email(request.email):
            raise ValueError(f"Email '{request.email}' already exists")

        user = User(
            username=request.username,
            email=request.email,
            full_name=request.full_name,
            age=request.age
        )

        return self._user_repository.save(user)

    def get_user_by_id(self, user_id: str) -> User:
        user = self._user_repository.find_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID '{user_id}' not found")
        return user

    def get_all_users(self) -> List[User]:
        return self._user_repository.find_all()

    def update_user(self, user_id: str, request: UpdateUserRequest) -> User:
        user = self.get_user_by_id(user_id)

        if request.email and request.email != user.email:
            if self._user_repository.exists_by_email(request.email):
                raise ValueError(f"Email '{request.email}' already exists")

        update_data = {k: v for k, v in request.dict().items() if v is not None}
        user.update(**update_data)

        return self._user_repository.save(user)

    def delete_user(self, user_id: str) -> bool:
        if not self._user_repository.find_by_id(user_id):
            raise ValueError(f"User with ID '{user_id}' not found")

        return self._user_repository.delete_by_id(user_id)

    def search_users_by_status(self, status: UserStatus) -> List[User]:
        all_users = self._user_repository.find_all()
        return [user for user in all_users if user.status == status]


# ===============================
# APPROACH 1: Single API Class with All Endpoints
# ===============================

class UserAPIController:
    """
    All FastAPI endpoints consolidated into a single class
    Similar to Spring Boot @RestController
    """

    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.router = APIRouter(prefix="/api/users", tags=["users"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup all routes for this controller"""
        self.router.add_api_route("/", self.create_user, methods=["POST"],
                                  response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
        self.router.add_api_route("/{user_id}", self.get_user, methods=["GET"],
                                  response_model=ApiResponse)
        self.router.add_api_route("/", self.get_all_users, methods=["GET"],
                                  response_model=ApiResponse)
        self.router.add_api_route("/{user_id}", self.update_user, methods=["PUT"],
                                  response_model=ApiResponse)
        self.router.add_api_route("/{user_id}", self.delete_user, methods=["DELETE"],
                                  response_model=ApiResponse)
        self.router.add_api_route("/search/status/{status}", self.search_users_by_status,
                                  methods=["GET"], response_model=ApiResponse)

    def create_user(self, request: CreateUserRequest) -> ApiResponse:
        """Create user endpoint"""
        try:
            user = self.user_service.create_user(request)
            return ApiResponse(
                success=True,
                message="User created successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Internal server error")

    def get_user(self, user_id: str) -> ApiResponse:
        """Get user by ID endpoint"""
        try:
            user = self.user_service.get_user_by_id(user_id)
            return ApiResponse(
                success=True,
                message="User retrieved successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    def get_all_users(self) -> ApiResponse:
        """Get all users endpoint"""
        users = self.user_service.get_all_users()
        return ApiResponse(
            success=True,
            message=f"Retrieved {len(users)} users",
            data=[user.to_dict() for user in users]
        )

    def update_user(self, user_id: str, request: UpdateUserRequest) -> ApiResponse:
        """Update user endpoint"""
        try:
            user = self.user_service.update_user(user_id, request)
            return ApiResponse(
                success=True,
                message="User updated successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            status_code = (status.HTTP_404_NOT_FOUND
                           if "not found" in str(e).lower()
                           else status.HTTP_400_BAD_REQUEST)
            raise HTTPException(status_code=status_code, detail=str(e))

    def delete_user(self, user_id: str) -> ApiResponse:
        """Delete user endpoint"""
        try:
            self.user_service.delete_user(user_id)
            return ApiResponse(
                success=True,
                message="User deleted successfully"
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    def search_users_by_status(self, status: UserStatus) -> ApiResponse:
        """Search users by status endpoint"""
        users = self.user_service.search_users_by_status(status)
        return ApiResponse(
            success=True,
            message=f"Found {len(users)} users with status '{status}'",
            data=[user.to_dict() for user in users]
        )


# ===============================
# APPROACH 2: Complete Application Class
# ===============================

class UserManagementApplication:
    """
    Complete FastAPI application encapsulated in a single class
    Similar to Spring Boot Application class
    """

    def __init__(self):
        # Initialize dependencies
        self.user_repository = InMemoryUserRepository()
        self.user_service = UserService(self.user_repository)

        # Initialize FastAPI app
        self.app = FastAPI(
            title="User Management API (Class-based)",
            description="FastAPI application consolidated into a single class",
            version="1.0.0"
        )

        # Setup routes
        self._setup_routes()
        self._setup_exception_handlers()

    def _setup_routes(self):
        """Setup all application routes"""
        # Health check
        self.app.add_api_route("/", self.health_check, methods=["GET"],
                               response_model=ApiResponse)

        # User endpoints
        self.app.add_api_route("/api/users", self.create_user, methods=["POST"],
                               response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
        self.app.add_api_route("/api/users/{user_id}", self.get_user, methods=["GET"],
                               response_model=ApiResponse)
        self.app.add_api_route("/api/users", self.get_all_users, methods=["GET"],
                               response_model=ApiResponse)
        self.app.add_api_route("/api/users/{user_id}", self.update_user, methods=["PUT"],
                               response_model=ApiResponse)
        self.app.add_api_route("/api/users/{user_id}", self.delete_user, methods=["DELETE"],
                               response_model=ApiResponse)
        self.app.add_api_route("/api/users/search/status/{status}", self.search_users_by_status,
                               methods=["GET"], response_model=ApiResponse)

    def _setup_exception_handlers(self):
        """Setup global exception handlers"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc: HTTPException):
            return ApiResponse(
                success=False,
                message=exc.detail,
                timestamp=datetime.now()
            )

    # API Endpoints
    async def health_check(self) -> ApiResponse:
        """Health check endpoint"""
        return ApiResponse(
            success=True,
            message="User Management API is running"
        )

    async def create_user(self, request: CreateUserRequest) -> ApiResponse:
        """Create user endpoint"""
        try:
            user = self.user_service.create_user(request)
            return ApiResponse(
                success=True,
                message="User created successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def get_user(self, user_id: str) -> ApiResponse:
        """Get user by ID endpoint"""
        try:
            user = self.user_service.get_user_by_id(user_id)
            return ApiResponse(
                success=True,
                message="User retrieved successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    async def get_all_users(self) -> ApiResponse:
        """Get all users endpoint"""
        users = self.user_service.get_all_users()
        return ApiResponse(
            success=True,
            message=f"Retrieved {len(users)} users",
            data=[user.to_dict() for user in users]
        )

    async def update_user(self, user_id: str, request: UpdateUserRequest) -> ApiResponse:
        """Update user endpoint"""
        try:
            user = self.user_service.update_user(user_id, request)
            return ApiResponse(
                success=True,
                message="User updated successfully",
                data=user.to_dict()
            )
        except ValueError as e:
            status_code = (status.HTTP_404_NOT_FOUND
                           if "not found" in str(e).lower()
                           else status.HTTP_400_BAD_REQUEST)
            raise HTTPException(status_code=status_code, detail=str(e))

    async def delete_user(self, user_id: str) -> ApiResponse:
        """Delete user endpoint"""
        try:
            self.user_service.delete_user(user_id)
            return ApiResponse(
                success=True,
                message="User deleted successfully"
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    async def search_users_by_status(self, status: UserStatus) -> ApiResponse:
        """Search users by status endpoint"""
        users = self.user_service.search_users_by_status(status)
        return ApiResponse(
            success=True,
            message=f"Found {len(users)} users with status '{status}'",
            data=[user.to_dict() for user in users]
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the application"""
        uvicorn.run(self.app, host=host, port=port)


# ===============================
# APPROACH 3: Using APIRouter with Class
# ===============================

def create_app_with_router():
    """Create FastAPI app using APIRouter approach"""
    # Initialize dependencies
    user_repository = InMemoryUserRepository()
    user_service = UserService(user_repository)

    # Create controller with router
    user_controller = UserAPIController(user_service)

    # Create FastAPI app
    app = FastAPI(
        title="User Management API (Router-based)",
        description="FastAPI with APIRouter consolidated in class",
        version="1.0.0"
    )

    # Include router
    app.include_router(user_controller.router)

    # Add health check
    @app.get("/", response_model=ApiResponse)
    async def health_check():
        return ApiResponse(
            success=True,
            message="User Management API is running (Router-based)"
        )

    return app


# ===============================
# Example Usage
# ===============================

if __name__ == "__main__":
    print("Choose an approach:")
    print("1. Complete Application Class")
    print("2. APIRouter with Controller Class")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("Starting Complete Application Class approach...")
        application = UserManagementApplication()
        application.run()

    elif choice == "2":
        print("Starting APIRouter approach...")
        app = create_app_with_router()
        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        print("Invalid choice. Starting default (Complete Application Class)...")
        application = UserManagementApplication()
        application.run()