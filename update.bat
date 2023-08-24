git pull origin main
CALL venv\Scripts\activate
python -m pip uninstall -y dataloader hsc103controller
python -m pip install git+https://github.com/PlusF/DataLoader
python -m pip install git+https://github.com/PlusF/HSC103Controller