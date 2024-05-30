import ttkbootstrap as ttk
from .main import STABox
import os
from pathlib import Path


PATH = Path(__file__).parent / 'assets'
app = ttk.Window("STABox")
app.option_add('*Font', 'Arial')
app.iconphoto(False, ttk.PhotoImage(file=os.path.join(PATH, 'app_icon.png')))
STABox(app)
app.mainloop()
