
$ install python library form rquirement.txt
pip install -r requirements.txt

$ How to run Fastapi

1.Activate the virtual environment

On Windows:
fastapi-env\Scripts\activate

On macOS and Linux:
source fastapi-env/bin/activate

2.Type in terminal 
uvicorn main:app --reload
to run server

$ Before commit run 
pip freeze > requirements.txt
