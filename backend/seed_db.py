from app.database import SessionLocal, engine, Base
from app.models import Tree

# Create tables if not already created
Base.metadata.create_all(bind=engine)

# Sample data to insert
sample_trees = [
    {
        "tree_id":"A-014", "zone":"Zone A", "severity":"high", "risk":"Canker suspected",
        "confidence":0.92, "last_seen":datetime(2026, 2, 10, 12, 30), "x":18, "y":22,
        "trend":"+12% risk", "notes":"Active lesions along trunk"
    },
    {
        "tree_id":"C-102", "zone":"Zone C", "severity":"high", "risk":"Severe drought stress",
        "confidence":0.88, "last_seen":datetime(2026, 2, 10, 9, 0), "x":62, "y":28,
        "trend":"+18% risk", "notes":"Canopy thinning and leaf curl observed"
    },
    {
        "tree_id":"B-033", "zone":"Zone B", "severity":"moderate", "risk":"Nutrient deficiency",
        "confidence":0.81, "last_seen":datetime(2026, 2, 10, 8, 0), "x":38, "y":55,
        "trend":"+5% risk", "notes":"Chlorosis pattern suggests nitrogen deficiency"
    },
    # Add more trees here as needed...
]

def seed():
    db: Session = SessionLocal()
    try:
        # Optional: clear existing table first
        db.query(models.Tree).delete()
        db.commit()

        for t in sample_trees:
            tree = models.Tree(**t)
            db.add(tree)
        db.commit()
        print(f"Inserted {len(sample_trees)} sample trees.")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
