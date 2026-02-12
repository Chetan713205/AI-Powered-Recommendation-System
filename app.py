from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from dotenv import load_dotenv
from pathlib import Path

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

from flipkart.data_ingestion import data_ingestion
from flipkart.rag_chain import RAGChainBuilder
from flipkart.config import Config

REQUEST_COUNT=Counter("http_requests_total", "Total HTTP requests")


def create_app():
    app = Flask(__name__)
    if not Config.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add OPENROUTER_API_KEY=your_key to your .env file. "
            "Get a key at https://openrouter.ai/keys"
        )
    vector_store = data_ingestion().ingest(load_existing=True)
    rag_chain=RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()      ## increnet by 1 enerytine homepage app is directed
        return render_template("index.html")
    
    @app.route("/get", methods=["POST"])
    def get_response():
        user_input=request.form["msg"]
        response=rag_chain.invoke(
            {"input" : user_input},
            config={"configurable" : {"session_id" : "user_session"}}
        )["answer"]
        return response
    
    @app.route("/metrics")
    def get_metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app
    
if __name__=="__main__":
    app=create_app()
    app.run(host="0.0.0.0", port=9000, debug=True)