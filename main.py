from fastapi import FastAPI
# Assuming your chat_routes.py is in api/routes
from api.routes import chat_routes 

# Create the main FastAPI application instance
app = FastAPI()

# Include your chat router
app.include_router(chat_routes.router)

# You might include other routers or middleware here later

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}