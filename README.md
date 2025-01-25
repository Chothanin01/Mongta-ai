$ Always activate Virtual Environment 

Create Virtual Environment
python -m venv venv-mongta

Activate by
On Windows:
venv-mongta\Scripts\activate

On macOS and Linux:
source fastapi-env/bin/activate

$ Install python library form rquirement.txt
pip install -r requirements.txt

$ How to run

Type in terminal 
uvicorn main:app --reload


$ Before commit run 
pip freeze > requirements.txt
