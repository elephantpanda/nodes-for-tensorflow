import wx

class AboutDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Frame.__init__(self, None, wx.ID_ANY, "About")
        self.SetInitialSize((300,200))
        self.SetPosition((400,200))
        self.SetBackgroundColour(wx.Colour(128,255,128))
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        text =wx.StaticText(self, 
            label="Nodes For Tesorflow\n\nVersion 0.1\n\nProgrammed by Paul Bird\n\nBased on tensorflow.",
            style=wx.ALIGN_CENTER)
        button = wx.Button(self, label="OK")
        button.Bind(wx.EVT_BUTTON, self.OK)
        vbox.Add(text,-1,wx.CENTER)
        vbox.Add(button,-1,wx.CENTER)
        self.Bind(wx.EVT_CLOSE, self.OK)

    def OK(self,event):
        self.Destroy()
