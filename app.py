import wx
import cv2


class Example(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)

        self.InitApp()

    def InitApp(self):
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        self.SetMenuBar(menubar)

        menubar.Append(fileMenu, '&File')
        fileOption = fileMenu.Append(wx.ID_EXIT, '&Exit', 'Exit the application')
        fileOption = fileMenu.Append(wx.ID_OPEN, '&Open', 'Select video')
        # fileMenu.AppendSeparator()

        self.Bind(wx.EVT_MENU, self.OnQuit, fileOption)
        self.Bind(wx.EVT_MENU, self.OnOpen, fileOption)

        self.SetSize(300, 200)
        self.SetTitle("simple menu")
        self.Centre()

    def OnQuit(self, event):
        self.Close()

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open video file", wildcard="(*.mp4;*.avi;*.mov)|*.mp4;*.avi;*.mov",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path_to_video = fileDialog.GetPath()
            cam = cv2.VideoCapture(path_to_video)
            ret, frame = cam.read()

            if ret:
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame)
                load_coords_dialog = wx.MessageDialog(None, "Load coordinates?", "Load coordinates",
                                                      wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
                ret = load_coords_dialog.ShowModal()
                if ret == wx.ID_YES:
                    load_coords_dialog.Destroy()
                else:
                    load_coords_dialog.Destroy()
                    self.ChooseCoordinates(cam)
            else:
                print("Error opening video file")

    def ChooseCoordinates(self, cam):
        coordinates = []
        while len(coordinates) < 4:
            ret, frame = cam.read()
            frame_copy = frame.copy()
            for i, coord in enumerate(coordinates):
                cv2.circle(frame_copy, coord, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame_copy, coordinates[i - 1], coord, (0, 255, 0), 2)
            if len(coordinates) == 4:
                cv2.line(frame_copy, coordinates[0], coordinates[3], (0, 255, 0), 2)
                cv2.line(frame_copy, coordinates[2], coordinates[3], (0, 255, 0), 2)
                cv2.line(frame_copy, coordinates[0], coordinates[1], (0, 255, 0), 2)
            cv2.imshow('frame', frame_copy)
            cv2.setMouseCallback('frame', self.OnClick, coordinates)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("Coordinates:", coordinates)

    def OnClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates = param
            if len(coordinates) == 3:
                coordinates.append((x, y))
            else:
                coordinates[-1] = (x, y)


def main():
    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
