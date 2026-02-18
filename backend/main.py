from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary in-memory data
trees = [
    {
        "id": 1,
        "tree_id": "A-014",
        "zone": "Zone A",
        "severity": "high",
        "risk": "Canker suspected",
        "confidence": 0.92,
        "last_seen": "2026-02-10T12:30:00Z",
        "notes": "Active lesions along trunk",
        "x": 18,
        "y": 22,
        "trend": "+12% risk"
    }
]

@app.get("/")
def root():
    return {"message": "Sylva backend running"}

@app.get("/trees")
def get_trees():
    return trees
