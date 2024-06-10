import wx
import cv2
import json
import os
from tracking import start_tracking

class Example(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)
        self.InitApp()
        self.coordinates = []
        self.real_life_coords = []
        self.video_path = ""
        self.frame = None  # Initialize frame to None

    def InitApp(self):
        panel = wx.Panel(self)

        button_size = (150, 50)  # Define button size to fit text and icons

        openVideoButton = wx.Button(panel, label='Open Video', pos=(10, 10), size=button_size)
        selectCoordsButton = wx.Button(panel, label='Select Coordinates', pos=(170, 10), size=button_size)
        loadCoordsButton = wx.Button(panel, label='Load Coordinates', pos=(330, 10), size=button_size)
        startTrackingButton = wx.Button(panel, label='Start Tracking', pos=(490, 10), size=button_size)
        quitButton = wx.Button(panel, label='Quit', pos=(650, 10), size=button_size)

        # Using built-in icons for buttons
        openVideoButton.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_BUTTON))
        selectCoordsButton.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_BUTTON))
        loadCoordsButton.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_EXECUTABLE_FILE, wx.ART_BUTTON))
        startTrackingButton.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_BUTTON))
        quitButton.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_QUIT, wx.ART_BUTTON))

        openVideoButton.Bind(wx.EVT_BUTTON, self.OnOpen)
        selectCoordsButton.Bind(wx.EVT_BUTTON, self.OnSelectCoords)
        loadCoordsButton.Bind(wx.EVT_BUTTON, self.OnLoadCoords)
        startTrackingButton.Bind(wx.EVT_BUTTON, self.OnStartTracking)
        quitButton.Bind(wx.EVT_BUTTON, self.OnQuit)

        self.selectCoordsButton = selectCoordsButton
        self.loadCoordsButton = loadCoordsButton
        self.startTrackingButton = startTrackingButton

        self.selectCoordsButton.Disable()
        self.loadCoordsButton.Disable()
        self.startTrackingButton.Disable()

        self.SetSize(800, 100)
        self.SetTitle("Object Tracking Application")
        self.Centre()

    def OnQuit(self, event):
        self.Close()

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open video file", wildcard="(*.mp4;*.avi;*.mov)|*.mp4;*.avi;*.mov",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            self.video_path = fileDialog.GetPath()
            self.ShowVideo(self.video_path)
            self.selectCoordsButton.Enable()
            self.loadCoordsButton.Enable()
            self.startTrackingButton.Enable()

    def ShowVideo(self, path_to_video):
        self.cam = cv2.VideoCapture(path_to_video)
        cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            cv2.imshow('video_frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key to close the window
                break
        self.cam.release()
        cv2.destroyAllWindows()

    def OnSelectCoords(self, event):
        self.cam = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cam.read()
        if ret:
            self.coordinates = []
            self.real_life_coords = []
            cv2.namedWindow('select_frame', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('select_frame', self.OnClick, self.coordinates)
            self.ShowFrame()
            self.ChooseCoordinates()
        self.cam.release()

    def OnLoadCoords(self, event):
        with wx.FileDialog(self, "Open coordinates file", wildcard="(*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path_to_coords = fileDialog.GetPath()
            self.LoadCoordinates(path_to_coords)
            self.LoadVideoFrame()
            self.ShowFrame()

    def LoadCoordinates(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
            self.coordinates = data['coordinates']
            self.real_life_coords = data['real_life_coords']
        print("Loaded coordinates:", self.coordinates)
        print("Loaded real-life coordinates:", self.real_life_coords)

    def SaveCoordinates(self, path):
        data = {
            'coordinates': self.coordinates,
            'real_life_coords': self.real_life_coords
        }
        with open(path, 'w') as file:
            json.dump(data, file)
        print("Coordinates saved to", path)

    def ShowFrame(self):
        while True:
            frame_copy = self.frame.copy()
            for i, coord in enumerate(self.coordinates):
                cv2.circle(frame_copy, tuple(coord), 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame_copy, tuple(self.coordinates[i - 1]), tuple(coord), (0, 255, 0), 2)
            if len(self.coordinates) == 4:
                cv2.line(frame_copy, tuple(self.coordinates[0]), tuple(self.coordinates[3]), (0, 255, 0), 2)
            cv2.imshow('select_frame', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key to close the window
                break
        cv2.destroyAllWindows()

    def LoadVideoFrame(self):
        if not self.video_path:
            wx.MessageBox('No video path specified.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        self.cam = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cam.read()
        if not ret:
            wx.MessageBox('Unable to read video frame.', 'Error', wx.OK | wx.ICON_ERROR)
        self.cam.release()

    def ChooseCoordinates(self):
        while len(self.coordinates) < 4:
            self.ShowFrame()
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key to close the window
                return
        print("Coordinates:", self.coordinates)
        print("Real-life Coordinates:", self.real_life_coords)
        if len(self.real_life_coords) < 4:
            self.coordinates = []
            self.real_life_coords = []
            wx.MessageBox('You cancelled entering coordinates. Please enter all 4 coordinates again.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        save_dialog = wx.FileDialog(self, "Save coordinates", wildcard="(*.json)|*.json",
                                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                    defaultFile=os.path.splitext(os.path.basename(self.video_path))[0] + ".json")
        if save_dialog.ShowModal() == wx.ID_OK:
            self.SaveCoordinates(save_dialog.GetPath())
        self.ShowFrame()

    def OnClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.coordinates) < 4:
                self.coordinates.append((x, y))
                real_x, real_y = self.PromptForRealCoordinates()
                if real_x is None or real_y is None:
                    self.coordinates = []
                    self.real_life_coords = []
                    wx.MessageBox('You cancelled entering coordinates. Please enter all 4 coordinates again.', 'Error', wx.OK | wx.ICON_ERROR)
                    return
                self.real_life_coords.append((real_x, real_y))
            self.ShowFrame()

    def PromptForRealCoordinates(self):
        dlg = wx.TextEntryDialog(self, "Enter the real-life X coordinate (default=0):", "Real-life X Coordinate", "0")
        if dlg.ShowModal() == wx.ID_OK:
            real_x = dlg.GetValue()
        else:
            dlg.Destroy()
            return None, None
        dlg.Destroy()

        dlg = wx.TextEntryDialog(self, "Enter the real-life Y coordinate (default=0):", "Real-life Y Coordinate", "0")
        if dlg.ShowModal() == wx.ID_OK:
            real_y = dlg.GetValue()
        else:
            dlg.Destroy()
            return None, None
        dlg.Destroy()

        return float(real_x), float(real_y)

    def OnStartTracking(self, event):
        start_tracking(self.coordinates, self.real_life_coords, self.video_path)
        self.EnableButtons(True)

    def EnableButtons(self, state):
        self.selectCoordsButton.Enable(state)
        self.loadCoordsButton.Enable(state)
        self.startTrackingButton.Enable(state)
        self.selectCoordsButton.Refresh()
        self.loadCoordsButton.Refresh()
        self.startTrackingButton.Refresh()

def main():

    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
