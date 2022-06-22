import networkx as nx
import numpy as np
import wx
from matplotlib import pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from PIL import Image
from graph import Graph


class GraphApp(wx.App):
    def __init__(self):
        super().__init__(clearSigInt=True)
        self.InitFrame()

        # init frame
    def InitFrame(self):
        frame = MainFrame(parent=None, title="Graph Recognition App", pos=(100, 100))
        frame.Show()
        return True


class MainFrame(wx.Frame):
    def __init__(self, parent, title, pos):
        super().__init__(parent=parent, title=title, pos=pos)
        self.OnInit()

    def OnInit(self):
        panel = GraphPanel(parent=self)
        self.SetSize(1200, 600)


class GraphPanel(wx.Panel):
    graph = None
    G = None

    def __init__(self, parent):
        super(GraphPanel, self).__init__(parent=parent)

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.toolbar = NavigationToolbar(self.canvas)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)

        # Inputs
        imagePathLabel = wx.StaticText(self, label="Image Path: ")
        self._imagePath = wx.TextCtrl(parent=self, value="")
        singlePathLabel = wx.StaticText(self, label="Single Path: ")
        self._singlePath = wx.TextCtrl(parent=self, value="")

        sourceVertexLabel = wx.StaticText(self, label="Source Vertex: ")
        self._source = wx.TextCtrl(parent=self, value="")
        targetVertexLabel = wx.StaticText(self, label="Target Vertex: ")
        self._target = wx.TextCtrl(parent=self, value="")

        removeSource = wx.StaticText(self, label="Source Vertex: ")
        self._remSource = wx.TextCtrl(parent=self, value="")
        removeTarget = wx.StaticText(self, label="Target Vertex: ")
        self._remTarget = wx.TextCtrl(parent=self, value="")

        removeVertex = wx.StaticText(self, label="Vertex ID: ")
        self._vertexID = wx.TextCtrl(parent=self, value="")

        # Button
        self._build = wx.Button(parent=self, label="Build Graph")
        self._build.Bind(event=wx.EVT_BUTTON, handler=self.buildGraph)

        self._add_edge = wx.Button(self, label="Add Edge")
        self._add_edge.Bind(event=wx.EVT_BUTTON, handler=self.addEdge)

        self._removeEdge = wx.Button(self, label="Remove Edge")
        self._removeEdge.Bind(event=wx.EVT_BUTTON, handler=self.removeEdge)

        self._export_tex = wx.Button(self, label="Export to LaTeX")
        self._export_tex.Bind(event=wx.EVT_BUTTON, handler=self.exportTEX)

        self._add_vertex = wx.Button(self, label="Add Vertex")
        self._add_vertex.Bind(event=wx.EVT_BUTTON, handler=self.addVertex)

        self._remove_vertex = wx.Button(self, label="Remove Vertex")
        self._remove_vertex.Bind(event=wx.EVT_BUTTON, handler=self.removeVertex)

        self._export_csv = wx.Button(self, label="Export to CSV")
        self._export_csv.Bind(event=wx.EVT_BUTTON, handler=self.exportAdjacency)

        self._isfilled = wx.CheckBox(self, label="Are the vertices filled?")

        # flash message

        self._message = wx.StaticText(self, label="")

        mainSizer = wx.BoxSizer()
        inputSizer = wx.BoxSizer(wx.VERTICAL)
        buttonSizer = wx.BoxSizer()
        buildSizer = wx.BoxSizer()

        buttonSizer.Add(self._add_vertex, 0, wx.ALL, 5)
        buttonSizer.Add(self._export_csv, 0, wx.ALL, 5)
        buttonSizer.Add(self._export_tex, 0, wx.ALL, 5)

        imagePathSizer = wx.BoxSizer()
        imagePathSizer.Add(imagePathLabel, 0, wx.ALL, 5)
        imagePathSizer.Add(self._imagePath, 1, wx.ALL | wx.EXPAND, 5)

        singlePathSizer = wx.BoxSizer()
        singlePathSizer.Add(singlePathLabel, 0, wx.ALL, 5)
        singlePathSizer.Add(self._singlePath, 1, wx.ALL | wx.EXPAND, 5)

        buildSizer.Add(window=self._isfilled, proportion=1, flag=wx.ALL | wx.CENTER, border=5)
        buildSizer.Add(window=self._build, proportion=1, flag=wx.ALL | wx.CENTER, border=5)

        addEdgeSizer = wx.BoxSizer()
        addEdgeSizer.Add(sourceVertexLabel, 0, wx.ALL, 5)
        addEdgeSizer.Add(self._source, 0, wx.ALL, 5)
        addEdgeSizer.Add(targetVertexLabel, 0, wx.ALL, 5)
        addEdgeSizer.Add(self._target, 0, wx.ALL, 5)
        addEdgeSizer.Add(self._add_edge, 0, wx.ALL, 5)

        removeEdgeSizer = wx.BoxSizer()
        removeEdgeSizer.Add(removeSource, 0, wx.ALL, 5)
        removeEdgeSizer.Add(self._remSource, 0, wx.ALL, 5)
        removeEdgeSizer.Add(removeTarget, 0, wx.ALL, 5)
        removeEdgeSizer.Add(self._remTarget, 0, wx.ALL, 5)
        removeEdgeSizer.Add(self._removeEdge, 0, wx.ALL, 5)

        removeVertexSizer = wx.BoxSizer()
        removeVertexSizer.Add(removeVertex, 0, wx.ALL, 5)
        removeVertexSizer.Add(self._vertexID, 0, wx.ALL, 5)
        removeVertexSizer.Add(self._remove_vertex, 0, wx.ALL, 5)

        # displaySizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        inputSizer.Add(imagePathSizer, 1, wx.EXPAND, 5)
        inputSizer.Add(singlePathSizer, 1, wx.EXPAND, 5)
        inputSizer.Add(buildSizer, proportion=1, flag=wx.ALL | wx.CENTER, border=5)
        inputSizer.Add(addEdgeSizer, 1, wx.EXPAND, 5)
        inputSizer.Add(removeEdgeSizer, 1, wx.EXPAND, 5)
        inputSizer.Add(removeVertexSizer, 1, wx.EXPAND, 5)

        inputSizer.Add(buttonSizer, 1, wx.CENTER, 5)
        inputSizer.Add(self._message, 1, wx.ALL | wx.EXPAND, 5)

        mainSizer.Add(self.vbox)
        mainSizer.Add(inputSizer, wx.EXPAND)

        self.SetSizer(mainSizer)
        self.Fit()

    def buildGraph(self, event):
        if self.graph is not None:
            return None

        image_path = self._imagePath.GetValue()
        single_path = self._singlePath.GetValue()
        filled = self._isfilled.GetValue()

        if image_path == "" or single_path == "":
            return None

        try:
            image, image_array = load_images(image_path)
            single, single_array = load_images(single_path)

        except FileNotFoundError:
            self.flashMessage("No file found!")
            return None

        try:
            self.graph = Graph(image_array, single_array, filled)
            self.showGraph()
        except ValueError:
            self.flashMessage("Error!")
            return None

    def showGraph(self):
        if self.graph is None:
            return None

        self.canvas.figure.clear()
        self.G = self.graph.nx_graph
        pos = {}
        for i in range(len(self.graph.vertices)):
            if i >= len(self.graph.vertices):
                x, y, r = self.graph.vertices[i-1].get_data()
                pass
            else:
                v = self.graph.vertices[i]
                x, y, r = v.get_data()
            pos[i] = (x, y * -1)

        nx.draw_networkx_nodes(self.G, pos=pos, node_color='r', node_size=400, alpha=1)
        nx.draw_networkx_labels(self.G, pos=pos)
        ax = plt.gca()
        for e in self.G.edges:
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="-", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])),
                                        ),
                        )
        plt.axis('on')
        self.canvas.draw()
        self.canvas.Refresh()

    def addEdge(self, event):
        if self.graph is None:
            return None

        source = self._source.GetValue()
        target = self._target.GetValue()

        if source == "" or target == "":
            return None

        self.graph.add_edge(int(source), int(target))
        self.showGraph()

    def addVertex(self, event):
        if self.graph is None:
            return None

        if self.graph.add_vertex():
            # Flash message
            self.flashMessage("Vertex added!")
        else:
            # You cannot add any new vertices
            self.flashMessage("No more vertices can be added!")
        self.showGraph()

    def exportTEX(self, event):
        if self.graph is None:
            return None
        self.graph.show_nx()
        self.flashMessage("Graph in LaTeX format saved!")

    def exportAdjacency(self, event):
        if self.graph is None:
            return None
        self.graph.export()
        self.flashMessage("Adjacency matrix saved!")

    def flashMessage(self, message):
        self._message.SetLabel(message)

    def removeVertex(self, event):
        if self.graph is None:
            return None
        vertexID = self._vertexID.GetValue()
        self.graph.remove_vertex(int(vertexID))
        self.showGraph()

    def removeEdge(self, event):
        if self.graph is None:
            return None

        source = self._remSource.GetValue()
        target = self._remTarget.GetValue()

        if source == "" or target == "":
            return None

        self.graph.remove_edge(int(source), int(target))
        self.showGraph()


def load_images(path_to_image):
    img = Image.open(path_to_image).convert("L")
    img_array = np.asarray(img)

    return img, img_array


if __name__ == '__main__':
    app = GraphApp()
    app.MainLoop()
