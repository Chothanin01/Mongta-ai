from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Get example
@app.get("/")
def read_root():
    return {"Hello": "World"}

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

# Post example
@app.post("/items/")
def create_item(item: Item):
    return item

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}