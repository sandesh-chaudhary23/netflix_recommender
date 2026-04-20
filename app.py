"""
CineAI v4  |  app.py
Netflix-style Movie Recommendation System
Cloud Project  |  Streamlit + Python + AWS
"""
 
import os, re, pickle, requests, base64, io
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
 
# ══════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineAI – Movie Discovery",
    page_icon="🎬", layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── CONFIG ────────────────────────────────────────────────────────────────────────
TMDB_KEY      = os.getenv("TMDB_API_KEY", "")
OMDB_KEY      = os.getenv("OMDB_API_KEY", "")          # free at omdbapi.com
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"      # public CDN, no key needed
 
ALL_GENRES = [
    "Action","Comedy","Drama","Romance","Science Fiction",
    "Horror","Thriller","Animation","Adventure","Crime",
    "Fantasy","Mystery","Documentary","History","Music",
]
GENRE_MAP = {
    "Action":          ["action","fight","war","battle","hero","mission","combat"],
    "Comedy":          ["comedy","funny","laugh","humor","hilarious","satire"],
    "Drama":           ["drama","life","family","emotion","story","struggle"],
    "Romance":         ["romance","love","heart","passion","relationship"],
    "Science Fiction": ["science fiction","sci-fi","space","alien","future","robot"],
    "Horror":          ["horror","scary","ghost","dark","fear","monster"],
    "Thriller":        ["thriller","suspense","danger","chase","conspiracy","espionage"],
    "Animation":       ["animation","animated","cartoon","anime","pixar","disney"],
    "Adventure":       ["adventure","journey","quest","explore","expedition"],
    "Crime":           ["crime","heist","gangster","mafia","detective","murder"],
    "Fantasy":         ["fantasy","magic","wizard","dragon","realm","myth"],
    "Mystery":         ["mystery","detective","secret","clue","puzzle"],
    "Documentary":     ["documentary","true story","history","biography"],
    "History":         ["history","historical","period","ancient","medieval","empire"],
    "Music":           ["music","musician","band","concert","singing","jazz","rock"],
}
 
# ══════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_data():
    for f in ["movies_clean.csv","similarity.pkl"]:
        if not Path(f).exists():
            st.error(f"❌ {f} not found. Run `python model.py` first.")
            st.stop()
    mv = pd.read_csv("movies_clean.csv")
    mv["title"]        = mv["title"].astype(str).str.strip()
    mv["vote_average"] = pd.to_numeric(mv.get("vote_average",0), errors="coerce").fillna(0)
    mv["popularity"]   = pd.to_numeric(mv.get("popularity",0),   errors="coerce").fillna(0)
    mv["vote_count"]   = pd.to_numeric(mv.get("vote_count",0),   errors="coerce").fillna(0)
    mv["runtime"]      = pd.to_numeric(mv.get("runtime",0),      errors="coerce").fillna(0)
    mv["year"]         = mv.get("year", pd.Series([0]*len(mv))).fillna(0).astype(int)
    for col in ["genres_str","overview","tagline","cast_str","director_str","poster_path","keywords_str"]:
        mv[col] = mv.get(col, pd.Series([""]*len(mv))).fillna("")
    with open("similarity.pkl","rb") as f:
        sim = pickle.load(f)
    return mv.reset_index(drop=True), sim
 
movies, similarity = load_data()
title_lower_map = {t.lower(): t for t in movies["title"]}
 
# ══════════════════════════════════════════════════════════════════════════════════
# POSTER ENGINE  (3-tier fallback — always shows something)
# ══════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=604800)
def fetch_tmdb_poster(title: str, year: int = 0) -> str:
    if not TMDB_KEY: return ""
    try:
        p = {"api_key": TMDB_KEY, "query": title, "language":"en-US"}
        if year: p["year"] = year
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=p, timeout=5)
        for res in r.json().get("results",[]):
            if res.get("poster_path"):
                return TMDB_IMG_BASE + res["poster_path"]
    except: pass
    return ""
 
@st.cache_data(show_spinner=False, ttl=604800)
def fetch_omdb_poster(title: str, year: int = 0) -> str:
    """OMDb API — free tier 1000 req/day, no domain needed, just API key."""
    if not OMDB_KEY: return ""
    try:
        p = {"apikey": OMDB_KEY, "t": title, "type": "movie"}
        if year: p["y"] = year
        r = requests.get("https://www.omdbapi.com/", params=p, timeout=5)
        poster = r.json().get("Poster","")
        if poster and poster != "N/A":
            return poster
    except: pass
    return ""
 
def _svg_poster(title: str, genres: str, score: float) -> str:
    """SVG fallback — works 100% offline, looks clean."""
    colors = {
        "action":"#8B0000","comedy":"#8B4500","drama":"#1a3a5c",
        "romance":"#5c1a4a","science fiction":"#0a3d3d","horror":"#1a1a1a",
        "thriller":"#1a2a1a","animation":"#5c3a00","adventure":"#1a4a1a",
        "crime":"#2a2a2a","fantasy":"#3a0a5c","mystery":"#0a1a2a",
        "documentary":"#2a3a2a","history":"#3a2a1a","music":"#3a003a",
    }
    g1 = (genres.lower().split()[0] if genres.strip() else "drama")
    color = colors.get(g1, "#1a1a2e")
    short = title[:18].replace("'","").replace('"','')
    score_s = f"★ {score:.1f}" if score else "★ N/A"
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='342' height='513'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='{color}'/>
      <stop offset='100%' stop-color='#050508'/>
    </linearGradient>
    <linearGradient id='sh' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='rgba(0,0,0,0)'/>
      <stop offset='100%' stop-color='rgba(0,0,0,0.9)'/>
    </linearGradient>
  </defs>
  <rect width='342' height='513' fill='url(#g)'/>
  <rect width='342' height='513' fill='url(#sh)'/>
  <text x='171' y='200' font-family='Georgia,serif' font-size='72'
        fill='rgba(229,9,20,0.5)' text-anchor='middle'>🎬</text>
  <rect x='20' y='380' width='302' height='1' fill='rgba(229,9,20,0.4)'/>
  <text x='171' y='420' font-family='Arial Black,Arial,sans-serif' font-size='15'
        font-weight='bold' fill='#ffffff' text-anchor='middle'>{short}</text>
  <text x='171' y='448' font-family='Arial,sans-serif' font-size='14'
        fill='#f5c518' text-anchor='middle'>{score_s}</text>
  <text x='171' y='472' font-family='Arial,sans-serif' font-size='11'
        fill='rgba(255,255,255,0.45)' text-anchor='middle'>{genres[:24]}</text>
</svg>"""
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"
 
def get_poster(row) -> str:
    title = str(row.get("title",""))
    year  = int(row.get("year",0) or 0)
 
    # Tier 1: TMDB CDN from dataset poster_path (FREE, no key)
    pp = str(row.get("poster_path","") or "").strip()
    if pp and pp not in ("nan","None","") and pp.startswith("/"):
        return TMDB_IMG_BASE + pp
 
    # Tier 2a: Live TMDB API (needs TMDB_API_KEY)
    if TMDB_KEY:
        url = fetch_tmdb_poster(title, year)
        if url: return url
 
    # Tier 2b: OMDb API (free, needs OMDB_API_KEY, no domain restriction)
    if OMDB_KEY:
        url = fetch_omdb_poster(title, year)
        if url: return url
 
    # Tier 3: SVG fallback (always works)
    return _svg_poster(title, str(row.get("genres_str","") or ""),
                       float(row.get("vote_average",0) or 0))
 
# ══════════════════════════════════════════════════════════════════════════════════
# ML FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════════
def recommend(title: str, n: int = 16) -> list:
    mask = movies["title"].str.lower() == title.lower()
    if not mask.any(): return []
    idx = movies[mask].index[0]
    row = similarity[idx]
    if hasattr(row,"toarray"): row = row.toarray().flatten()
    else: row = np.asarray(row).flatten()
    pairs = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[1:n+1]
    return [movies.iloc[i].to_dict() for i,_ in pairs]
 
def fuzzy_search(q: str, limit: int = 12) -> list:
    q = q.strip().lower()
    if len(q) < 2: return []
    exact    = [t for tl,t in title_lower_map.items() if tl == q]
    starts   = [t for tl,t in title_lower_map.items() if tl.startswith(q) and tl != q]
    contains = [t for tl,t in title_lower_map.items() if q in tl and not tl.startswith(q)]
    return (exact + starts + contains)[:limit]
 
def get_trending(n=16):    return movies.nlargest(n,"popularity").to_dict("records")
def get_top_rated(n=16):   return movies[movies["vote_count"]>=200].nlargest(n,"vote_average").to_dict("records")
def get_new_releases(n=16):return movies[movies["year"]>2010].nlargest(n,"year").to_dict("records")
def get_hidden_gems(n=16):
    sub = movies[(movies["vote_count"]>=50)&(movies["vote_average"]>=7.5)]
    return sub.nsmallest(n,"popularity").head(n).to_dict("records")
def get_by_genre(genre: str, n=16) -> list:
    kws  = GENRE_MAP.get(genre,[genre.lower()])
    mask = movies["genres_str"].str.lower().apply(lambda g: any(k in g for k in kws))
    sub  = movies[mask]
    if len(sub)<n:
        extra = movies[~mask].sample(min(n-len(sub),max(0,len(movies)-len(sub))),random_state=42)
        sub   = pd.concat([sub,extra])
    return sub.nlargest(n,"popularity").head(n).to_dict("records")
def get_by_decade(decade: int, n=12):
    sub = movies[(movies["year"]>=decade)&(movies["year"]<decade+10)]
    return sub.nlargest(n,"vote_average").to_dict("records") if len(sub) else []
def star_rating(s: float) -> str:
    f = min(5,max(0,round(s/2)))
    return "★"*f+"☆"*(5-f)
 
# ══════════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');
:root{--red:#e50914;--rd:rgba(229,9,20,.18);--gold:#f5c518;--bg:#0a0a0f;--bg2:#141418;--bg3:#1c1c22;--txt:#e8e8ec;--dim:rgba(255,255,255,.38);--border:rgba(255,255,255,.07);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:var(--bg)!important;color:var(--txt);font-family:'Inter',sans-serif;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]>div{padding:0!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--red);border-radius:3px;}
 
/* NAVBAR */
.navbar{position:fixed;top:0;left:0;right:0;z-index:9999;display:flex;align-items:center;justify-content:space-between;padding:0 3.5rem;height:62px;background:linear-gradient(180deg,rgba(0,0,0,.98) 0%,transparent 100%);backdrop-filter:blur(14px);border-bottom:1px solid var(--border);}
.nb-logo{font-family:'Bebas Neue',cursive;font-size:2rem;letter-spacing:.14em;color:var(--red);text-shadow:0 0 40px rgba(229,9,20,.65);}
.nb-sub{font-size:.62rem;font-weight:300;color:rgba(255,255,255,.28);letter-spacing:.25em;margin-top:-5px;}
.nb-links{display:flex;gap:2rem;}
.nb-link{font-size:.82rem;font-weight:500;color:rgba(255,255,255,.5);text-decoration:none;transition:color .2s;}
.nb-link:hover{color:#fff;}
.nb-tag{font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--red);background:var(--rd);border:1px solid rgba(229,9,20,.35);padding:.2rem .55rem;border-radius:2px;}
 
/* HERO */
.hero{position:relative;width:100%;height:80vh;min-height:480px;display:flex;align-items:flex-end;padding:0 3.5rem 4rem;overflow:hidden;margin-top:62px;background:radial-gradient(ellipse 100% 70% at 50% -5%,#200a0a 0%,var(--bg) 65%);}
.hero-glow{position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse 55% 65% at 75% 20%,rgba(229,9,20,.14) 0%,transparent 70%),radial-gradient(ellipse 35% 40% at 12% 78%,rgba(80,0,150,.08) 0%,transparent 70%);}
.hero-reel{position:absolute;right:-3%;top:0;bottom:0;width:52%;display:grid;grid-template-columns:repeat(7,1fr);gap:2px;opacity:.07;transform:skewX(-4deg);overflow:hidden;pointer-events:none;}
.hero-cell{background:var(--red);animation:rpulse 5s ease-in-out infinite;}
.hero-cell:nth-child(3n){background:#7a0000;animation-delay:1.5s;}
.hero-cell:nth-child(7n){background:#ff4444;animation-delay:3s;}
@keyframes rpulse{0%,100%{opacity:1}50%{opacity:.2}}
.hero-content{position:relative;z-index:2;max-width:580px;}
.hero-eyebrow{display:inline-flex;align-items:center;gap:.45rem;font-size:.62rem;font-weight:700;letter-spacing:.24em;text-transform:uppercase;color:var(--red);background:var(--rd);border:1px solid rgba(229,9,20,.3);padding:.22rem .7rem;border-radius:2px;margin-bottom:.9rem;}
.hero-title{font-family:'Bebas Neue',cursive;font-size:clamp(3.2rem,6.5vw,5.5rem);line-height:.92;letter-spacing:.02em;color:#fff;text-shadow:0 10px 80px rgba(0,0,0,.98);margin-bottom:.95rem;}
.hero-title em{color:var(--red);font-style:normal;}
.hero-desc{font-size:.9rem;font-weight:300;line-height:1.72;color:rgba(255,255,255,.43);margin-bottom:2rem;}
.hero-stats{display:flex;gap:2.8rem;}
.stat-n{font-family:'Bebas Neue',cursive;font-size:1.7rem;color:var(--red);line-height:1;}
.stat-l{font-size:.6rem;color:rgba(255,255,255,.26);letter-spacing:.14em;text-transform:uppercase;}
 
/* SEARCH */
.search-section{padding:1.6rem 3.5rem 1.4rem;background:rgba(0,0,0,.45);border-top:1px solid var(--border);border-bottom:1px solid var(--border);}
.search-label{font-family:'Bebas Neue',cursive;font-size:.9rem;letter-spacing:.22em;color:rgba(255,255,255,.28);margin-bottom:.55rem;}
 
/* Widget overrides */
div[data-testid="stTextInput"] input{background:rgba(255,255,255,.07)!important;border:1.5px solid rgba(255,255,255,.1)!important;border-radius:5px!important;color:#fff!important;font-family:'Inter',sans-serif!important;font-size:.95rem!important;padding:.72rem 1rem!important;caret-color:var(--red);}
div[data-testid="stTextInput"] input:focus{border-color:rgba(229,9,20,.55)!important;box-shadow:0 0 0 3px rgba(229,9,20,.1)!important;outline:none!important;}
div[data-testid="stTextInput"]>label{display:none!important;}
div[data-testid="stSelectbox"]>div>div{background:rgba(255,255,255,.07)!important;border:1.5px solid rgba(255,255,255,.1)!important;border-radius:5px!important;color:#fff!important;font-family:'Inter',sans-serif!important;font-size:.88rem!important;}
div[data-testid="stSelectbox"] svg{fill:rgba(255,255,255,.3)!important;}
div[data-testid="stSelectbox"] label{color:rgba(255,255,255,.38)!important;font-size:.75rem!important;}
div[data-testid="stTabs"] button{font-family:'Bebas Neue',cursive!important;letter-spacing:.12em!important;font-size:.9rem!important;color:rgba(255,255,255,.45)!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#fff!important;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--red)!important;}
div[data-testid="stTabs"] [data-baseweb="tab-border"]{background:var(--border)!important;}
.stButton>button{background:var(--red)!important;color:#fff!important;border:none!important;border-radius:4px!important;font-family:'Inter',sans-serif!important;font-size:.85rem!important;font-weight:700!important;letter-spacing:.07em!important;padding:.72rem 1.6rem!important;width:100%!important;transition:background .2s,transform .15s!important;}
.stButton>button:hover{background:#b81d24!important;transform:translateY(-1px)!important;}
 
/* REC BANNER */
.rec-banner{margin:1.5rem 3.5rem;padding:1.4rem 2rem;background:linear-gradient(135deg,rgba(229,9,20,.1),rgba(80,0,30,.08));border:1px solid rgba(229,9,20,.22);border-radius:7px;display:flex;align-items:center;gap:1.4rem;}
.rec-icon{font-size:2rem;}
.rec-title{font-family:'Bebas Neue',cursive;font-size:1.15rem;color:#fff;letter-spacing:.06em;}
.rec-sub{font-size:.76rem;color:rgba(255,255,255,.36);margin-top:.1rem;}
 
/* ROW */
.row-section{padding:1.6rem 3.5rem;}
.row-hd{display:flex;align-items:baseline;gap:.75rem;margin-bottom:1rem;}
.row-ttl{font-family:'Bebas Neue',cursive;font-size:1.3rem;letter-spacing:.07em;color:#fff;}
.pill{font-size:.58rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;padding:.15rem .5rem;border-radius:2px;}
.pill-red{background:rgba(229,9,20,.18);color:var(--red);border:1px solid rgba(229,9,20,.35);}
.pill-gold{background:rgba(245,197,24,.13);color:var(--gold);border:1px solid rgba(245,197,24,.3);}
.pill-blue{background:rgba(0,145,255,.13);color:#0091ff;border:1px solid rgba(0,145,255,.28);}
.pill-teal{background:rgba(0,200,160,.13);color:#00c8a0;border:1px solid rgba(0,200,160,.28);}
.pill-purple{background:rgba(160,0,255,.13);color:#a000ff;border:1px solid rgba(160,0,255,.28);}
.pill-pink{background:rgba(255,60,130,.13);color:#ff3c82;border:1px solid rgba(255,60,130,.28);}
 
/* MOVIE GRID */
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:.9rem;}
.mcard{position:relative;border-radius:6px;overflow:hidden;background:var(--bg3);cursor:pointer;aspect-ratio:2/3;transition:transform .3s cubic-bezier(.34,1.56,.64,1),box-shadow .3s;}
.mcard:hover{transform:scale(1.08) translateY(-7px);box-shadow:0 30px 80px rgba(0,0,0,.92),0 0 0 2px rgba(229,9,20,.45);z-index:30;}
.mcard img{width:100%;height:100%;object-fit:cover;display:block;background:var(--bg3);transition:filter .3s;}
.mcard:hover img{filter:brightness(.45);}
.mcard-overlay{position:absolute;inset:0;background:linear-gradient(0deg,rgba(0,0,0,.97) 0%,rgba(0,0,0,.15) 55%,transparent 100%);display:flex;flex-direction:column;justify-content:flex-end;padding:.7rem .65rem;opacity:0;transition:opacity .3s;}
.mcard:hover .mcard-overlay{opacity:1;}
.mcard-play{position:absolute;top:50%;left:50%;transform:translate(-50%,-55%) scale(.7);width:44px;height:44px;border-radius:50%;background:rgba(229,9,20,.85);display:flex;align-items:center;justify-content:center;font-size:1.1rem;opacity:0;transition:opacity .3s,transform .3s;}
.mcard:hover .mcard-play{opacity:1;transform:translate(-50%,-55%) scale(1);}
.mcard-title{font-size:.75rem;font-weight:600;color:#fff;line-height:1.25;margin-bottom:.16rem;}
.mcard-stars{font-size:.66rem;color:var(--gold);margin-top:.08rem;}
.mcard-meta{font-size:.62rem;color:rgba(255,255,255,.4);margin-top:.1rem;}
.mcard-genre{display:inline-block;margin-top:.18rem;font-size:.58rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--red);background:rgba(229,9,20,.15);padding:.05rem .3rem;border-radius:2px;}
.mcard-badge{position:absolute;top:.4rem;left:.4rem;font-family:'Bebas Neue',cursive;font-size:.78rem;color:#fff;background:rgba(229,9,20,.88);padding:.07rem .32rem;border-radius:2px;}
.mcard-score{position:absolute;top:.4rem;right:.4rem;font-size:.64rem;font-weight:700;background:rgba(0,0,0,.8);color:var(--gold);padding:.07rem .3rem;border-radius:2px;}
.mcard-new{position:absolute;top:.4rem;right:.4rem;font-size:.58rem;font-weight:700;background:#00c850;color:#000;padding:.07rem .3rem;border-radius:2px;letter-spacing:.06em;}
 
/* DETAIL CARD */
.detail-card{margin:1.5rem 3.5rem;padding:2rem;background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid var(--border);border-radius:8px;display:flex;gap:2rem;}
.detail-poster{flex-shrink:0;width:160px;border-radius:5px;overflow:hidden;}
.detail-poster img{width:100%;display:block;}
.detail-title{font-family:'Bebas Neue',cursive;font-size:1.6rem;color:#fff;letter-spacing:.04em;margin-bottom:.3rem;}
.detail-tagline{font-size:.82rem;font-style:italic;color:rgba(255,255,255,.35);margin-bottom:.8rem;}
.detail-meta{display:flex;gap:1.5rem;margin-bottom:.9rem;flex-wrap:wrap;}
.detail-meta-item{font-size:.75rem;color:rgba(255,255,255,.45);}
.detail-meta-item b{color:#fff;font-weight:600;}
.detail-overview{font-size:.84rem;line-height:1.7;color:rgba(255,255,255,.55);}
.detail-cast{font-size:.76rem;color:rgba(255,255,255,.4);margin-top:.6rem;}
.detail-cast b{color:rgba(255,255,255,.65);}
 
/* GENRE TABS */
.g-tabs{display:flex;gap:.45rem;flex-wrap:wrap;margin-bottom:1rem;}
.g-tab{font-size:.7rem;font-weight:500;padding:.3rem .85rem;border-radius:3px;border:1px solid var(--border);background:rgba(255,255,255,.04);color:rgba(255,255,255,.48);}
.g-tab.active{background:rgba(229,9,20,.16);border-color:rgba(229,9,20,.42);color:var(--red);}
 
/* DIVIDER */
.hdiv{height:1px;margin:0 3.5rem;background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent);}
 
/* FOOTER */
.footer{margin-top:4rem;padding:2rem 3.5rem;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;}
.f-logo{font-family:'Bebas Neue',cursive;font-size:1.3rem;color:var(--red);letter-spacing:.1em;}
.f-links{display:flex;gap:1.5rem;}
.f-link{font-size:.72rem;color:rgba(255,255,255,.3);text-decoration:none;}
.f-link:hover{color:rgba(255,255,255,.6);}
.f-txt{font-size:.68rem;color:rgba(255,255,255,.18);}
</style>
""", unsafe_allow_html=True)
 
# ── SESSION STATE ─────────────────────────────────────────────────────────────────
for k,v in dict(sel_movie=None,sel_genre="Action",query="",sugg=[],recs=[],detail=None,decade=2000).items():
    if k not in st.session_state: st.session_state[k]=v
 
# ── NAVBAR ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div><div class="nb-logo">🎬 CineAI</div><div class="nb-sub">INTELLIGENT MOVIE DISCOVERY</div></div>
  <div class="nb-links">
    <a class="nb-link" href="#">Home</a>
    <a class="nb-link" href="#">Trending</a>
    <a class="nb-link" href="#">Top Rated</a>
    <a class="nb-link" href="#">Genres</a>
  </div>
  <span class="nb-tag">ML POWERED</span>
</div>
""", unsafe_allow_html=True)
 
# ── HERO ──────────────────────────────────────────────────────────────────────────
cells="".join(['<div class="hero-cell"></div>']*84)
st.markdown(f"""
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-reel">{cells}</div>
  <div class="hero-content">
    <div class="hero-eyebrow">🧠 TF-IDF · Cosine Similarity · TMDB Dataset</div>
    <div class="hero-title">DISCOVER<br>YOUR NEXT<br><em>OBSESSION.</em></div>
    <div class="hero-desc">Real ML engine · {len(movies):,} movies · Partial search · Instant recommendations</div>
    <div class="hero-stats">
      <div><div class="stat-n">{len(movies):,}</div><div class="stat-l">Movies</div></div>
      <div><div class="stat-n">{len(ALL_GENRES)}</div><div class="stat-l">Genres</div></div>
      <div><div class="stat-n">8K</div><div class="stat-l">Features</div></div>
      <div><div class="stat-n">AWS</div><div class="stat-l">Cloud</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── SEARCH ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="search-section"><div class="search-label">🔍 SEARCH ANY MOVIE · GET INSTANT AI RECOMMENDATIONS</div>', unsafe_allow_html=True)
c1,c2 = st.columns([5,1])
with c1:
    raw_q = st.text_input("_", value=st.session_state.query,
        placeholder="Type movie name — e.g.  spider,  inception,  dark knight …",
        label_visibility="collapsed", key="qbox")
with c2:
    st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
    go = st.button("🔍 Search")
 
if raw_q != st.session_state.query:
    st.session_state.query = raw_q
    st.session_state.sugg  = fuzzy_search(raw_q) if len(raw_q.strip())>=2 else []
if go and len(raw_q.strip())>=2:
    st.session_state.sugg = fuzzy_search(raw_q)
 
if st.session_state.sugg:
    opts   = ["— select to get recommendations —"]+st.session_state.sugg
    picked = st.selectbox("Search Results:", opts, key="pick_box")
    if picked != opts[0] and picked != st.session_state.sel_movie:
        st.session_state.sel_movie = picked
        st.session_state.recs      = recommend(picked,16)
        m = movies[movies["title"].str.lower()==picked.lower()]
        if len(m): st.session_state.detail = m.iloc[0].to_dict()
elif raw_q.strip() and len(raw_q.strip())>=2 and go:
    st.markdown(f'<p style="color:rgba(255,255,255,.35);font-size:.82rem;padding:.4rem 0">No results for "<b style="color:#e50914">{raw_q}</b>". Try: <b>spider</b>, <b>batman</b>, <b>avatar</b>, <b>toy story</b>.</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
 
# ── CARD RENDERER ─────────────────────────────────────────────────────────────────
def render_row(records:list, badge_fn=None, show_new=False):
    if not records: return
    cards=""
    for i,row in enumerate(records):
        poster  = get_poster(row)
        score   = float(row.get("vote_average",0) or 0)
        year    = int(row.get("year",0) or 0)
        title   = str(row["title"])
        short   = title[:22]+("…" if len(title)>22 else "")
        genres  = str(row.get("genres_str","") or "").split()
        g_tag   = genres[0] if genres else ""
        runtime = int(row.get("runtime",0) or 0)
        badge   = badge_fn(i) if badge_fn else ""
        sc_b    = f'<div class="mcard-score">★ {score:.1f}</div>' if score and not show_new else ""
        new_b   = '<div class="mcard-new">NEW</div>' if show_new and not badge_fn else ""
        meta    = " · ".join(p for p in [str(year) if year else "",f"{runtime}m" if runtime else ""] if p)
        cards  += f"""
        <div class="mcard">
          <img src="{poster}" alt="{title}" loading="lazy">
          <div class="mcard-play">▶</div>
          {badge}{new_b}{sc_b}
          <div class="mcard-overlay">
            <div class="mcard-title">{short}</div>
            <div class="mcard-stars">{star_rating(score)} {score:.1f}</div>
            <div class="mcard-meta">{meta}</div>
            <div class="mcard-genre">{g_tag}</div>
          </div>
        </div>"""
    st.markdown(f'<div class="mgrid">{cards}</div>', unsafe_allow_html=True)
 
# ── DETAIL CARD ───────────────────────────────────────────────────────────────────
if st.session_state.detail:
    d       = st.session_state.detail
    poster  = get_poster(d)
    score   = float(d.get("vote_average",0) or 0)
    year    = int(d.get("year",0) or 0)
    runtime = int(d.get("runtime",0) or 0)
    cast    = str(d.get("cast_str","") or "").replace("_"," ")[:120]
    director= str(d.get("director_str","") or "").replace("_"," ")
    tagline = str(d.get("tagline","") or "")
    overview= str(d.get("overview","") or "")[:380]
    genres  = str(d.get("genres_str","") or "")
    st.markdown(f"""
    <div class="detail-card">
      <div class="detail-poster"><img src="{poster}" alt="{d['title']}"></div>
      <div>
        <div class="detail-title">{d['title']}</div>
        {f'<div class="detail-tagline">"{tagline}"</div>' if tagline else ''}
        <div class="detail-meta">
          <div class="detail-meta-item"><b>⭐ {score:.1f}</b> / 10</div>
          {f'<div class="detail-meta-item"><b>{year}</b></div>' if year else ''}
          {f'<div class="detail-meta-item"><b>{runtime}m</b></div>' if runtime else ''}
          <div class="detail-meta-item">{genres}</div>
        </div>
        {f'<div class="detail-overview">{overview}…</div>' if overview else ''}
        {f'<div class="detail-cast"><b>Director:</b> {director}</div>' if director else ''}
        {f'<div class="detail-cast"><b>Cast:</b> {cast}</div>' if cast else ''}
      </div>
    </div>""", unsafe_allow_html=True)
 
# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────────
if st.session_state.sel_movie and st.session_state.recs:
    sel=st.session_state.sel_movie
    st.markdown(f"""
    <div class="rec-banner">
      <div class="rec-icon">🎯</div>
      <div>
        <div class="rec-title">Because You Selected: "{sel}"</div>
        <div class="rec-sub">ML matched {len(st.session_state.recs)} films · TF-IDF cosine similarity on genres · keywords · director · overview</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">🎯 Recommended For You</div><span class="pill pill-red">AI MATCH</span></div>', unsafe_allow_html=True)
    render_row(st.session_state.recs, badge_fn=lambda i: f'<div class="mcard-badge">#{i+1}</div>')
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
 
# ── TABS ──────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs(["🔥  TRENDING","⭐  TOP RATED","🎭  BY GENRE","🆕  NEW RELEASES","💎  HIDDEN GEMS"])
 
with tab1:
    st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">🔥 Trending Now</div><span class="pill pill-red">LIVE POPULARITY</span></div>', unsafe_allow_html=True)
    render_row(get_trending(16), badge_fn=lambda i: f'<div class="mcard-badge">#{i+1}</div>')
    st.markdown("</div>", unsafe_allow_html=True)
 
with tab2:
    st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">⭐ Critically Acclaimed</div><span class="pill pill-gold">≥200 VOTES</span></div>', unsafe_allow_html=True)
    render_row(get_top_rated(16))
    st.markdown("</div>", unsafe_allow_html=True)
 
with tab3:
    st.markdown('<div class="row-section">', unsafe_allow_html=True)
    tabs_html="".join(f'<div class="g-tab {"active" if g==st.session_state.sel_genre else ""}">{g}</div>' for g in ALL_GENRES)
    st.markdown(f'<div class="g-tabs">{tabs_html}</div>', unsafe_allow_html=True)
    sel_genre=st.selectbox("Genre",ALL_GENRES,index=ALL_GENRES.index(st.session_state.sel_genre),label_visibility="visible",key="genre_sel")
    st.session_state.sel_genre=sel_genre
    st.markdown(f'<div class="row-hd" style="margin-top:.8rem"><div class="row-ttl">🎭 {sel_genre}</div><span class="pill pill-blue">GENRE MATCH</span></div>', unsafe_allow_html=True)
    render_row(get_by_genre(sel_genre,16))
    st.markdown("</div>", unsafe_allow_html=True)
 
with tab4:
    st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">🆕 New Releases</div><span class="pill pill-teal">POST-2010</span></div>', unsafe_allow_html=True)
    render_row(get_new_releases(16),show_new=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
with tab5:
    st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">💎 Hidden Gems</div><span class="pill pill-purple">HIGH RATING · LOW HYPE</span></div><p style="font-size:.78rem;color:rgba(255,255,255,.35);margin-bottom:.9rem">Highly rated films most people haven\'t discovered yet.</p>', unsafe_allow_html=True)
    render_row(get_hidden_gems(16))
    st.markdown("</div>", unsafe_allow_html=True)
 
# ── BY DECADE ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
st.markdown('<div class="row-section"><div class="row-hd"><div class="row-ttl">🕰 Browse By Decade</div><span class="pill pill-pink">TIME TRAVEL</span></div>', unsafe_allow_html=True)
decades=[1970,1980,1990,2000,2010,2020]
dec_cols=st.columns(len(decades))
for col,d in zip(dec_cols,decades):
    with col:
        if st.button(f"{d}s",key=f"dec_{d}"): st.session_state.decade=d
dm=get_by_decade(st.session_state.decade,12)
if dm:
    st.markdown(f'<div class="row-hd" style="margin-top:.8rem"><div class="row-ttl">Best of the {st.session_state.decade}s</div></div>', unsafe_allow_html=True)
    render_row(dm)
st.markdown("</div>", unsafe_allow_html=True)
 
# ── FOOTER ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div><div class="f-logo">🎬 CineAI</div><div class="f-txt" style="margin-top:.3rem">Content-Based ML · TF-IDF · Cosine Similarity · AWS EC2</div></div>
  <div class="f-links">
    <a class="f-link" href="#">About</a>
    <a class="f-link" href="#">GitHub</a>
    <a class="f-link" href="#">AWS</a>
    <a class="f-link" href="#">API Docs</a>
  </div>
  <div class="f-txt">{len(movies):,} movies · TMDB Dataset · © 2025 CineAI · Cloud Project</div>
</div>
""", unsafe_allow_html=True)
 