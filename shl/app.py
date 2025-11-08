import os

import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="SHL Assessment Recommender")


class Req(BaseModel):
    job_title: str = ""
    description: str = ""
    url: str = ""
    top_k: int = 5


EMBED_PROVIDER = os.getenv("EMBED_API_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "textembedding-gecko-001")
MODELS_READY = False
df = None
embeddings = None
tf = None
tfidf_matrix = None


def load_models():
    """
    Load only precomputed artifacts: assessments_df.pkl, embeddings.npy (optional), tfidf.pkl, tfidf_matrix.pkl
    """
    global MODELS_READY, df, embeddings, tf, tfidf_matrix
    try:
        required = [
            "shl/models/assessments_df.pkl",
            "shl/models/tfidf.pkl",
            "shl/models/tfidf_matrix.pkl",
        ]
        for p in required:
            if not os.path.exists(p):
                MODELS_READY = False
                print(f"[load_models] missing: {p}")
                return

        df = joblib.load("shl/models/assessments_df.pkl")
        tf = joblib.load("shl/models/tfidf.pkl")
        tfidf_matrix = joblib.load("shl/models/tfidf_matrix.pkl")
        
        # embeddings are optional - load if available
        if os.path.exists("shl/models/embeddings.npy"):
            embeddings = np.load("shl/models/embeddings.npy")
            print("[load_models] models loaded successfully with embeddings")
        else:
            embeddings = None
            print("[load_models] models loaded successfully (TF-IDF only mode - no embeddings)")
        
        MODELS_READY = True
    except Exception as e:
        MODELS_READY = False
        print("[load_models] ERROR loading models:", str(e))


@app.on_event("startup")
def on_startup():
    # configure Gemini
    if EMBED_PROVIDER == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print("[startup] Gemini configured")
        else:
            print("[startup] GEMINI_API_KEY not set")
    load_models()


def fetch_url_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def normalize(values):
    array = np.array(values, dtype=float)
    return (array - array.min()) / (array.max() - array.min() + 1e-9)


def get_query_embedding_gemini(text: str) -> np.ndarray:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    # create embedding
    resp = genai.embeddings.create(model=GEMINI_EMBED_MODEL, input=text)
    # response structure: resp.data[0].embedding
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    return emb


def get_query_embedding(text: str) -> np.ndarray:
    if EMBED_PROVIDER == "gemini":
        return get_query_embedding_gemini(text)
    else:
        raise RuntimeError("Unsupported EMBED_PROVIDER. Use 'gemini'.")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>SHL Assessment Recommender</title>
      <style>
        :root{
          --bg:#f3f6fb;
          --card:#ffffff;
          --primary:#004b9f;    /* deep SHL-like blue */
          --accent:#0078d4;
          --muted:#556277;
          --radius:12px;
          --shadow: 0 8px 30px rgba(13,38,66,0.08);
        }
        *{box-sizing:border-box}
        body{
          margin:0;
          font-family: "Segoe UI", Roboto, -apple-system, "Helvetica Neue", Arial;
          background: linear-gradient(180deg,var(--bg),#eef5fc);
          color:#0b1220;
          -webkit-font-smoothing:antialiased;
        }
        .container{
          width:100%;
          max-width:1000px;
          margin:40px auto;
          padding:20px;
        }
        header{
          display:flex;
          gap:16px;
          align-items:center;
          margin-bottom:20px;
        }
        .brand {
          width:72px;
          height:72px;
          border-radius:14px;
          background:linear-gradient(135deg,var(--primary),var(--accent));
          display:flex;
          align-items:center;
          justify-content:center;
          color:white;
          font-weight:700;
          font-size:22px;
          box-shadow:var(--shadow);
        }
        .title h1{ margin:0; font-size:22px; letter-spacing: -0.2px; }
        .title p{ margin:6px 0 0; color:var(--muted); font-size:14px; }

        .grid{ display:grid; grid-template-columns: 1fr 380px; gap:20px; align-items:start; }
        .card{
          background:var(--card);
          border-radius:var(--radius);
          padding:22px;
          box-shadow:var(--shadow);
        }

        label{ display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
        textarea, input[type="text"], input[type="number"]{
          width:100%;
          padding:12px;
          border-radius:10px;
          border:1px solid #e7eef8;
          background:#fff;
          font-size:14px;
          color:#071427;
          resize:vertical;
        }
        .muted { color:var(--muted); font-size:13px; }

        .controls{ display:flex; gap:10px; margin-top:12px; }
        button.primary{
          background:var(--primary);
          color:white; border:none;
          padding:10px 14px; border-radius:10px; cursor:pointer;
          font-weight:600;
        }
        button.ghost{
          background:transparent; border:1px solid #dbeafc; color:var(--primary);
          padding:10px 12px; border-radius:10px; cursor:pointer;
        }

        .right{
          display:flex; flex-direction:column; gap:14px;
        }
        .meta{ background:linear-gradient(180deg,#f8fbff,#ffffff); padding:16px; border-radius:10px; border:1px solid #eef6ff; }
        .meta h4{ margin:0 0 6px 0; font-size:15px; }
        .meta p{ margin:0; color:var(--muted); font-size:13px; }

        .results{ margin-top:16px; }
        table{ width:100%; border-collapse:collapse; margin-top:10px; }
        th, td{ padding:10px 8px; text-align:left; border-bottom:1px solid #f3f6fb; font-size:14px; }
        th{ color:var(--muted); font-weight:600; font-size:13px; }

        .score{ color:var(--primary); font-weight:700; }

        footer{ margin-top:16px; text-align:center; color:var(--muted); font-size:13px; }

        @media (max-width:920px){
          .grid{ grid-template-columns: 1fr; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <div class="brand">SHL</div>
          <div class="title">
            <h1>SHL Assessment Recommender</h1>
            <p>Built according to the SHL specification and dataset — get recommended individual assessments for job descriptions.</p>
          </div>
        </header>

        <div class="grid">
          <div class="card">
            <h3 style="margin:0 0 10px 0">Try it live</h3>
            <form id="form" onsubmit="return false;">
              <div>
                <label for="job">Job title / description or JD URL</label>
                <textarea id="job" rows="5" placeholder="e.g. Senior QA Engineer - Selenium, SQL, Automation, stakeholder collaboration"></textarea>
                <div class="muted" style="margin-top:8px">Tip: you can paste a full job description or a link. If a URL is used, the service will fetch its text before matching.</div>
              </div>

              <div style="display:flex; gap:12px; margin-top:12px; align-items:center;">
                <div style="flex:1; min-width:120px;">
                  <label for="topk">Top K</label>
                  <input id="topk" type="number" value="5" min="1" max="10" />
                </div>
                <div style="display:flex; gap:8px; align-items:flex-end;">
                  <button class="primary" id="go">Recommend</button>
                  <button class="ghost" id="clear">Clear</button>
                </div>
              </div>
            </form>

            <div id="status" class="muted" style="margin-top:12px">Service: <a href="/health" class="muted">/health</a></div>

            <div id="out" class="results"></div>
          </div>

          <div class="right">
            <div class="meta card">
              <h4>How it works</h4>
              <p class="muted">This system combines TF-IDF lexical matching and Gemini semantic embeddings to rank SHL individual assessments. It balances technical and behavioural results when needed.</p>
            </div>

            <div class="meta card">
              <h4>Endpoints</h4>
              <p class="muted"><code>/health</code> — Health check (GET)<br/><code>/recommend</code> — Recommendation (POST JSON)</p>
            </div>

            <div class="meta card">
              <h4>Deployment</h4>
              <p class="muted">Deployed on Render. Use environment variables <code>GEMINI_API_KEY</code> and <code>EMBED_API_PROVIDER=gemini</code>.</p>
            </div>
          </div>
        </div>

        <footer>Made with FastAPI • Scikit-learn • Gemini API — <span class="muted">Rashul Rajput</span></footer>
      </div>

      <script>
        const go = document.getElementById('go');
        const clearBtn = document.getElementById('clear');
        const out = document.getElementById('out');
        const status = document.getElementById('status');

        clearBtn.addEventListener('click', ()=> {
          document.getElementById('job').value = '';
          document.getElementById('topk').value = 5;
          out.innerHTML = '';
        });

        go.addEventListener('click', async () => {
          out.innerHTML = '';
          const job = document.getElementById('job').value.trim();
          const topk = Number(document.getElementById('topk').value) || 5;
          if(!job){
            out.innerHTML = '<div class="muted">Please enter a job title, description or URL.</div>';
            return;
          }
          status.textContent = 'Processing...';
          go.disabled = true;
          clearBtn.disabled = true;

          try {
            const body = JSON.stringify({ job_title: job, description: "", url: "" , top_k: topk });
            const res = await fetch('/recommend', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body
            });
            if(!res.ok){
              const txt = await res.text();
              out.innerHTML = '<div style="color:#b00020">Error: ' + (txt || res.statusText) + '</div>';
              status.textContent = 'Error';
              return;
            }
            const data = await res.json();
            status.textContent = `Found ${data.recommendations?.length || 0} recommendations`;
            if(!data.recommendations || data.recommendations.length === 0){
              out.innerHTML = '<div class="muted">No recommendations found.</div>';
            } else {
              const rows = data.recommendations.map(r =>
                `<tr>
                  <td><a href="${escapeHtml(r.assessment_url)}" target="_blank" rel="noopener">${escapeHtml(r.assessment_name)}</a></td>
                  <td class="score">${(Math.round((r.score||0)*100)/100).toFixed(2)}</td>
                </tr>`
              ).join('');
              out.innerHTML = `<div style="overflow:auto"><table><thead><tr><th>Assessment</th><th>Score</th></tr></thead><tbody>${rows}</tbody></table></div>`;
            }
          } catch (err) {
            out.innerHTML = '<div style="color:#b00020">Request failed — check the service logs or network.</div>';
            status.textContent = 'Failed';
          } finally {
            go.disabled = false;
            clearBtn.disabled = false;
          }
        });

        function escapeHtml(s){
          if(!s) return '';
          return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
        }
      </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    status = "ok" if MODELS_READY else "starting"
    return {"status": status}


@app.post("/recommend")
def recommend(req: Req):
    try:
        if not any([req.job_title, req.description, req.url]):
            raise HTTPException(status_code=400, detail="No input provided")
        if req.top_k < 1 or req.top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
        if not MODELS_READY:
            raise HTTPException(status_code=503, detail="Models not ready yet. Please try again shortly.")

        # Build query text
        q_text = " ".join([req.job_title or "", req.description or ""]).strip()
        if req.url:
            q_text += " " + fetch_url_text(req.url)

        # 1) Query embedding via Gemini
        try:
            q_emb = get_query_embedding(q_text)
        except Exception as e:
            # If external API fails, fallback to TF-IDF only
            print("[recommend] embedding API failed:", str(e))
            q_emb = None

        # 2) Semantic similarity (if q_emb present and embeddings loaded)
        if q_emb is not None and embeddings is not None:
            sem = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]
            sem_n = normalize(sem)
        else:
            sem_n = np.zeros(len(df))

        # 3) TF-IDF similarity
        q_vec = tf.transform([q_text])
        tfidf_sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        tf_n = normalize(tfidf_sims)

        # 4) Combine hybrid score
        # adjust weights as needed
        final = 0.65 * sem_n + 0.35 * tf_n

        # 5) Balance K (Knowledge) and P (Personality) if both domains present
        tech_tokens = ["python","java","sql","javascript","react","node","c++","c#","docker","kubernetes","coding","technical","software","engineer","qa","selenium"]
        beh_tokens = ["collaborat","team","stakehold","communication","personality","behaviour","behavior","leadership","culture","interpersonal"]
        txt_low = q_text.lower()
        has_tech = any(t in txt_low for t in tech_tokens)
        has_beh = any(t in txt_low for t in beh_tokens)

        idx_sorted = np.argsort(final)[::-1]
        top_k = req.top_k

        selected_idx = []
        if has_tech and has_beh and top_k >= 2:
            need_k = max(1, top_k // 2)
            need_p = max(1, top_k - need_k)
            k_added = 0
            p_added = 0
            test_types = df.get("test_type", pd.Series([""]*len(df))).fillna("").astype(str).tolist()
            for i in idx_sorted:
                if len(selected_idx) >= top_k:
                    break
                t = test_types[i].upper() if i < len(test_types) else ""
                if "K" in t and k_added < need_k:
                    selected_idx.append(i); k_added += 1
                elif "P" in t and p_added < need_p:
                    selected_idx.append(i); p_added += 1
            for i in idx_sorted:
                if i not in selected_idx and len(selected_idx) < top_k:
                    selected_idx.append(i)
        else:
            selected_idx = list(idx_sorted[:top_k])

        out = []
        for i in selected_idx:
            out.append({
                "assessment_name": df.iloc[i]["title"],
                "assessment_url": df.iloc[i]["url"],
                "score": float(final[i])
            })
        return {"recommendations": out}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})