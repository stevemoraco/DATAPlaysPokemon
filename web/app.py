from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import logging
import json
from web.agent_runner import run_agent
from agent.simple_agent import SimpleAgent

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists for serving assets
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory="web/templates")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        if not self.active_connections:
            logger.warning("No active connections to broadcast to")
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)

manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Pass LLM provider and model to template for dynamic theming
    provider = getattr(request.app.state.args, 'provider', 'anthropic')
    model_name = getattr(request.app.state.args, 'model', None)
    context = {
        "request": request,
        "provider": provider,
        "model_name": model_name,
    }
    return templates.TemplateResponse(
        "index.html",
        context
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back the received data for testing
                await websocket.send_text(f"Server received: {data}")
            except WebSocketDisconnect:
                await manager.disconnect(websocket)
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    finally:
        await manager.disconnect(websocket)

# Function to send game state updates, with optional environment info
async def send_game_updates(frame_data: bytes, claude_message: str, env: str = None):
    try:
        message = {
            "type": "update",
            "frame": frame_data.hex(),  # Convert bytes to hex string
            "message": claude_message
        }
        if env is not None:
            message["env"] = env
        await manager.broadcast(json.dumps(message))
    except Exception as e:
        logger.error(f"Error sending game updates: {e}")

@app.post("/start")
async def start_agent():
    if app.state.agent_task is not None and not app.state.agent_task.done():
        return {"status": "error", "message": "Agent is already running"}
    
    try:
        # Reset the pause flag when starting
        app.state.is_paused = False
        
        # If we don't have an agent yet, create one
        if not hasattr(app.state, 'agent'):
            # Enable sound in emulator; audio streaming to client not yet supported
            app.state.agent = SimpleAgent(
                rom_path=app.state.args.rom_path,
                headless=True,
                sound=True,
                max_history=app.state.args.max_history,
                app=app
            )
        
        app.state.agent_task = asyncio.create_task(
            run_agent(
                agent=app.state.agent,
                num_steps=app.state.args.steps,
                run_log_dir=app.state.run_log_dir,
                send_game_updates=send_game_updates,
                claude_logger=app.state.claude_logger
            )
        )
        return {"status": "success", "message": "Agent started successfully"}
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/pause")
async def pause_agent():
    if not hasattr(app.state, 'agent_task') or app.state.agent_task is None:
        return {"status": "error", "message": "No agent is running"}
    
    try:
        # Toggle pause state
        app.state.is_paused = not getattr(app.state, 'is_paused', False)
        return {"status": "success", "message": "Agent pause state toggled"}
    except Exception as e:
        logger.error(f"Error toggling pause state: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/stop")
async def stop_agent():
    if not hasattr(app.state, 'agent_task') or app.state.agent_task is None:
        return {"status": "error", "message": "No agent is running"}
    
    try:
        # Cancel the running task
        app.state.agent_task.cancel()
        try:
            await app.state.agent_task
        except asyncio.CancelledError:
            pass
        
        # Reset state
        app.state.agent_task = None
        app.state.is_paused = False
        
        # Save logs if needed
        logger.info("Agent stopped, logs saved")
        
        return {"status": "success", "message": "Agent stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping agent: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/status")
async def get_agent_status():
    is_running = app.state.agent_task is not None and not app.state.agent_task.done()
    is_paused = getattr(app.state, 'is_paused', False)
    
    if not is_running:
        return {"status": "stopped"}
    elif is_paused:
        return {"status": "paused"}
    else:
        return {"status": "running"}

@app.post("/upload-save-state")
async def upload_save_state(file: UploadFile = File(...)):
    try:
        # Create saves directory if it doesn't exist
        saves_dir = Path("saves")
        saves_dir.mkdir(exist_ok=True)
        
        # Validate file extension
        if not file.filename.endswith('.state'):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "Invalid file type. Must be a PyBoy .state file."}
            )
        
        # Save the uploaded file
        save_path = saves_dir / file.filename
        with save_path.open("wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Load the save state into the emulator if agent exists
        if hasattr(app.state, 'agent'):
            try:
                app.state.agent.emulator.load_state(str(save_path))
                logger.info(f"Loaded save state from {save_path}")
                return JSONResponse({"status": "success", "message": "Save state loaded successfully"})
            except Exception as e:
                logger.error(f"Failed to load save state: {e}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"status": "error", "message": f"Failed to load save state: {str(e)}"}
                )
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "Agent not initialized"}
            )
            
    except Exception as e:
        logger.error(f"Error handling save state upload: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        ) 