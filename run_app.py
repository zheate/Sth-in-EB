import streamlit_desktop_app

def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Set up the environment
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Path to the streamlit app
    app_path = resolve_path(os.path.join("app", "app.py"))
    
    # Run the desktop app
    streamlit_desktop_app.start_desktop_app(app_path)
