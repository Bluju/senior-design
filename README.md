# Senior Design

## Set up instructions

This project uses a python virtual envioronment 
To set up your virtual environment, use these commands:

```
python -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
python -m pip install -r requirements.txt   
```

If you add a package to the project, use this command to update the requirements.txt file:

```
py -m pip freeze > requirements.txt
```

## Running the program
While the virtual environment is active:

```
python POMDP.py
```
