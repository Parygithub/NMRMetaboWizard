Installation
============

Supported environment
---------------------

- Python 3.10 or newer
- Windows, macOS, or Linux
- Chrome or Edge recommended

Create an environment
---------------------

Windows::

   python -m venv .venv
   .venv\Scripts\activate.bat
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

macOS/Linux::

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

Run the app
-----------

::

   python -m shiny run --reload --launch-browser app.py

Docker
------

::

   docker compose up --build

Then open ``http://localhost:8000``.
