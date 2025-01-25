$ Always Activate the virtual environment

create virtual environment
python -m venv fastapi-env

Activate by
On Windows:
fastapi-env\Scripts\activate

On macOS and Linux:
source fastapi-env/bin/activate

$ install python library form rquirement.txt
pip install -r requirements.txt


$ How to run Fastapi

Type in terminal 
uvicorn main:app --reload
to run server

$ Before commit run 
pip freeze > requirements.txt
