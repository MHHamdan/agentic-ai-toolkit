"""WebSocket connection manager for real-time updates."""
from fastapi import WebSocket
from typing import Dict, Set, List
import json
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.evaluation_subscribers: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from evaluation subscribers
        for subscribers in self.evaluation_subscribers.values():
            subscribers.discard(websocket)

    def subscribe_to_evaluation(self, evaluation_id: str, websocket: WebSocket):
        """Subscribe to updates for a specific evaluation."""
        if evaluation_id not in self.evaluation_subscribers:
            self.evaluation_subscribers[evaluation_id] = set()
        self.evaluation_subscribers[evaluation_id].add(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        message["timestamp"] = datetime.now().isoformat()
        text = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_to_evaluation(self, evaluation_id: str, message: dict):
        """Send message to subscribers of a specific evaluation."""
        if evaluation_id not in self.evaluation_subscribers:
            return

        message["timestamp"] = datetime.now().isoformat()
        message["evaluation_id"] = evaluation_id
        text = json.dumps(message)
        disconnected = []

        for connection in self.evaluation_subscribers[evaluation_id]:
            try:
                await connection.send_text(text)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific client."""
        message["timestamp"] = datetime.now().isoformat()
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


# Singleton instance
manager = ConnectionManager()
