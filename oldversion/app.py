
'''
class DirDialog(wx.Frame):


    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, -1, u"职位推荐系统")
        b = wx.Button(self, -1, u"加载评分数据")
        self.Bind(wx.EVT_BUTTON, self.OnButton, b)


    def OnButton(self, event):
        """"""
        dlg = wx.DirDialog(self, u"选择文件夹", style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            print dlg.GetPath()  # 文件夹路径

        dlg.Destroy()

if __name__ == '__main__':
    frame = wx.PySimpleApp()
    app = DirDialog()
    app.Show()
    frame.MainLoop()


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(200, 100))
        self.control = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.CreateStatusBar()  # A Statusbar in the bottom of the window

        # Setting up the menu.
        filemenu = wx.Menu()

        # wx.ID_ABOUT and wx.ID_EXIT are standard IDs provided by wxWidgets.
        filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        self.Show(True)


app = wx.App(False)
frame = MainWindow(None, "Sample editor")
app.MainLoop()  
'''