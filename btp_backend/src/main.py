from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from .llm import main2
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"hello": "world"}

class ConnectionManager:
    """Class defining socket events"""
    def __init__(self):
        """Init method, keeping track of connections"""
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        """Connect event"""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Disconnect event"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a direct message to the connected client"""
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/start")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)  # Parse incoming JSON string into a dictionary
            # gen="hi"
            gen=main2.generate(data['message'])
            print("gen",gen)
            if(gen==False):
                gen="Cant predict"
            response = {"content": gen}
            await manager.send_personal_message(response, websocket) 
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
