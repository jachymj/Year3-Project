import matplotlib.pyplot as plt
import networkx as nx
import wx
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


class NetworkFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1)
        self.panel = GraphPanel(self)


class GraphPanel(wx.Panel):
    def __init__(self, parent):
        super(GraphPanel, self).__init__(parent=parent)
        self.fig = plt.figure()
        self.canvas = FigCanvas(self, -1, self.fig)
        G = nx.house_graph()
        pos = {0: (0, 0),
               1: (1, 0),
               2: (0, 1),
               3: (1, 1),
               4: (0.5, 2.0)}

        nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[4])
        nx.draw_networkx_nodes(G, pos, node_size=3000, nodelist=[0, 1, 2, 3], node_color='b')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
        plt.axis('off')
        mainSizer = wx.BoxSizer()
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.toolbar = NavigationToolbar(self.canvas)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        mainSizer.Add(self.vbox)
        self.SetSizer(mainSizer)
        self.vbox.Fit(self)






#plt.savefig("house_with_colors.png") # save as png
#plt.show() # display

if __name__ == '__main__':
  app = wx.App()
  app.frame = NetworkFrame()
  app.frame.Show()
  app.MainLoop()