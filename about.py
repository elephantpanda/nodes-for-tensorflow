import Tkinter as tk

class AboutDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        top.geometry("300x200+200+100")
        tk.Label(top, text="Nodes For Tesorflow\n\nVersion 0.1\n\nProgrammed by Paul Bird\n\nBased on tensorflow.").pack()
        b = tk.Button(top, text="OK", command=self.ok)
        b.pack(pady=5)
    def ok(self):
        self.top.destroy()
