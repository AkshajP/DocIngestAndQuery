from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Assuming your chat_routes.py is in api/routes
from api.routes import chat_routes 
from api.routes import document_routes

# Create the main FastAPI application instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include your chat router
app.include_router(chat_routes.router)
app.include_router(document_routes.router)
# You might include other routers or middleware here later

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}