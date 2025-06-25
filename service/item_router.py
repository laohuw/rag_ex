# from fastapi import APIRouter
#
# class BaseRouter:
#     def __init__(self):
#         self.router = APIRouter()
#         self._register_routes()
#
#     def _register_routes(self):
#         raise NotImplementedError("Subclasses should implement this method")
#
#     @classmethod
#     def register_routes(cls, app):
#         instance = cls()
#         app.include_router(instance.router, prefix="/api", tags=[cls.__name__])
#
# class ItemRouter(BaseRouter):
#     def __init__(self, service: ItemService):
#         self.service = service
#         super().__init__(service)
#
#     def _register_routes(self):
#         @self.router.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
#         async def create_item(item: ItemCreate):
#             return await self.service.create_item(item)
#
#         @self.router.get("/items/{item_id}", response_model=Item)
#         async def read_item(item_id: int):
#             return await self.service.get_item(item_id)
#
#         @self.router.put("/items/{item_id}", response_model=Item)
#         async def update_item(item_id: int, item: Item):
#             return await self.service.update_item(item_id, item)
#
#         @self.router.delete("/items/{item_id}")
#         async def delete_item(item_id: int):
#             return await self.service.delete_item(item_id)