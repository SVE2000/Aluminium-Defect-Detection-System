# sender_cam_ws.py (30 FPS optimized)
import asyncio
import websockets
from picamera2 import Picamera2
import io
import traceback
import sys
import os


CONFIG_FILE = "receiver_ip.txt"  # Configuration file name
DEFAULT_IP = "192.168.1.109"  # Fallback IP if file doesn't exist
DEFAULT_PORT = 8765
FRAME_INTERVAL = 1 / 30  # ~30 FPS


def read_receiver_uri(config_file):
    """
    Read receiver IP and port from configuration file.
    
    File format:
        192.168.1.109:8765
    or just:
        192.168.1.109
    
    Returns WebSocket URI string.
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                line = f.read().strip()
                
            if not line:
                print(f"⚠️ Config file is empty. Using default: {DEFAULT_IP}:{DEFAULT_PORT}")
                return f"ws://{DEFAULT_IP}:{DEFAULT_PORT}"
            
            # Check if port is included
            if ':' in line:
                ip_port = line.split(':')
                ip = ip_port[0].strip()
                port = ip_port[1].strip()
                uri = f"ws://{ip}:{port}"
            else:
                ip = line.strip()
                uri = f"ws://{ip}:{DEFAULT_PORT}"
            
            print(f"✅ Loaded receiver URI from {config_file}: {uri}")
            return uri
        else:
            # Create config file with default IP
            with open(config_file, 'w') as f:
                f.write(f"{DEFAULT_IP}:{DEFAULT_PORT}")
            print(f"📝 Created {config_file} with default IP: {DEFAULT_IP}:{DEFAULT_PORT}")
            return f"ws://{DEFAULT_IP}:{DEFAULT_PORT}"
            
    except Exception as e:
        print(f"⚠️ Error reading config file: {e}. Using default: {DEFAULT_IP}:{DEFAULT_PORT}")
        return f"ws://{DEFAULT_IP}:{DEFAULT_PORT}"


async def send_frames():
    # Read receiver URI from config file
    RECEIVER_URI = read_receiver_uri(CONFIG_FILE)
    
    picam2 = Picamera2()

    # Use video configuration for higher FPS
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        buffer_count=3
    )
    picam2.configure(config)
    picam2.start()
    print("📸 Camera started at 30 FPS mode (640x480).")

    websocket = None
    retry_delay = 1

    try:
        while True:
            try:
                print(f"🔗 Connecting to {RECEIVER_URI} ...")
                websocket = await websockets.connect(RECEIVER_URI, max_size=None, close_timeout=5)
                print("✅ Connected to receiver!")
                retry_delay = 1

                while True:
                    try:
                        # Capture frame from video stream
                        stream = io.BytesIO()
                        picam2.capture_file(stream, format='jpeg')
                        image_data = stream.getvalue()

                        # Send frame
                        await websocket.send(image_data)
                        print(f"📤 Sent frame ({len(image_data)} bytes)")

                        await asyncio.sleep(FRAME_INTERVAL)

                    except (websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                            OSError) as e:
                        print(f"❌ Connection lost: {type(e).__name__}. Reconnecting...")
                        break
                    except Exception as e:
                        if "call_soon" not in str(e):
                            print(f"⚠️ Error during send: {e}")
                            traceback.print_exc()
                        break

            except (ConnectionRefusedError, OSError) as e:
                print(f"⚠️ Cannot connect to receiver: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay, 30)

            except asyncio.CancelledError:
                print("🛑 Task cancelled.")
                raise
            except Exception as e:
                if "call_soon" not in str(e):
                    print(f"⚠️ Unexpected error: {e}")
                    traceback.print_exc()
                await asyncio.sleep(retry_delay)

            if websocket is not None:
                try:
                    await websocket.close()
                except Exception:
                    pass
                websocket = None

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:
                pass
        picam2.close()
        print("📷 Camera stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(send_frames())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except RuntimeError as e:
        if "call_soon" in str(e) or "cannot schedule" in str(e):
            print("⚙️ Gracefully handled asyncio shutdown.")
            sys.exit(0)
        else:
            raise
