import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routers import instrumental_remover, dj_tag_remover

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.include_router(instrumental_remover.router, tags=["Instrumental Remover"])
app.include_router(dj_tag_remover.router, tags=["DJ Tag Remover"])


@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serves the main index page to choose between tools."""
    return templates.TemplateResponse("index.html", {"request": request})