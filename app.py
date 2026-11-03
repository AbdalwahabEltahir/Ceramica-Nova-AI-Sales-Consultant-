from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real env vars always take precedence

from backend.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
