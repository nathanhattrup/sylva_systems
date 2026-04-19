from database import Base, engine
from models import Tree

Base.metadata.create_all(bind=engine)
print("Database and tables created!")
