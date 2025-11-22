import os
import sys
import socket
import time
import threading
import webview
import streamlit_desktop_app
import logging

# Configure logging
log_file = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__), 'debug_log.txt')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        logging.info(f"Found free port: {port}")
        return port

def wait_for_server(port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                logging.info("Server is reachable")
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    logging.error("Timeout waiting for server")
    return False

if __name__ == "__main__":
    try:
        logging.info("Application starting...")
        
        # Set up the environment
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_THEME_BASE"] = "light"
        
        # Path to the streamlit app
        app_path = resolve_path(os.path.join("app", "app.py"))
        logging.info(f"App path: {app_path}")
        
        # Find a free port
        port = get_free_port()
        
        # Start Streamlit in a separate thread
        def start_streamlit():
            logging.info("Starting Streamlit server thread")
            try:
                # Use streamlit_desktop_app's helper to run streamlit
                # It handles the command line arguments construction
                streamlit_desktop_app.run_streamlit(app_path, {'server.port': str(port)})
            except Exception as e:
                logging.error(f"Streamlit thread error: {e}")

        t = threading.Thread(target=start_streamlit)
        t.daemon = True
        t.start()
        
        # Wait for Streamlit to start
        if wait_for_server(port):
            logging.info("Creating webview window")
            # Create window with zoom enabled
            webview.create_window(
                "数据分析工具",
                f"http://localhost:{port}",
                width=1400,
                height=900,
                zoomable=True  # This enables the zoom functionality
            )
            webview.start()
            logging.info("Webview closed")
        else:
            logging.error("Failed to start Streamlit server")
            print("Failed to start Streamlit server")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
