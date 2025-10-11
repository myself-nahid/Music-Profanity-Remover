from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routers import instrumental_remover, dj_tag_remover

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(instrumental_remover.router, tags=["Instrumental Remover"])
app.include_router(dj_tag_remover.router, tags=["DJ Tag Remover"])


@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serves the main index page to choose between tools."""
    return templates.TemplateResponse("index.html", {"request": request})