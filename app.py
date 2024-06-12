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
        self.detection_area = []
        self.additional_areas = []
        self.video_path = ""
        self.frame = None
        self.scaled_frame = None
        self.scale_factor_x = 1
        self.scale_factor_y = 1
        self.static_bitmap = None
        self.img_size = 1280  # Default image size
        self.confidence = 0.5  # Default confidence

    def InitApp(self):
        panel = wx.Panel(self)
        self.panel = panel

        button_size = (140, 40)

        openVideoButton = wx.Button(panel, label='Open Video', pos=(10, 10), size=button_size)
        selectCoordsButton = wx.Button(panel, label='Select Coordinates', pos=(160, 10), size=button_size)
        selectDetectionAreaButton = wx.Button(panel, label='Select Detection Area', pos=(310, 10), size=button_size)
        addAdditionalAreaButton = wx.Button(panel, label='Add Additional Area', pos=(460, 10), size=button_size)
        saveAreasButton = wx.Button(panel, label='Save Areas', pos=(610, 10), size=button_size)
        loadCoordsButton = wx.Button(panel, label='Load File', pos=(760, 10), size=button_size)
        startTrackingButton = wx.Button(panel, label='Start Tracking', pos=(910, 10), size=button_size)
        quitButton = wx.Button(panel, label='Quit', pos=(1060, 10), size=button_size)

        imgSizeLabel = wx.StaticText(panel, label="Image Size:", pos=(1210, 15))
        self.imgSizeChoice = wx.Choice(panel, choices=["320", "640", "1280"], pos=(1280, 10), size=(70, 30))
        self.imgSizeChoice.SetSelection(2)  # Default to 1280

        confLabel = wx.StaticText(panel, label="Confidence:", pos=(1360, 15))
        self.confChoice = wx.Choice(panel, choices=["0.3", "0.5", "0.7"], pos=(1440, 10), size=(70, 30))
        self.confChoice.SetSelection(1)  # Default to 0.5

        openVideoButton.Bind(wx.EVT_BUTTON, self.OnOpen)
        selectCoordsButton.Bind(wx.EVT_BUTTON, self.OnSelectCoords)
        selectDetectionAreaButton.Bind(wx.EVT_BUTTON, self.OnSelectDetectionArea)
        addAdditionalAreaButton.Bind(wx.EVT_BUTTON, self.OnAddAdditionalArea)
        saveAreasButton.Bind(wx.EVT_BUTTON, self.OnSaveAreas)
        loadCoordsButton.Bind(wx.EVT_BUTTON, self.OnLoadCoords)
        startTrackingButton.Bind(wx.EVT_BUTTON, self.OnStartTracking)
        quitButton.Bind(wx.EVT_BUTTON, self.OnQuit)

        self.selectCoordsButton = selectCoordsButton
        self.selectDetectionAreaButton = selectDetectionAreaButton
        self.addAdditionalAreaButton = addAdditionalAreaButton
        self.saveAreasButton = saveAreasButton
        self.loadCoordsButton = loadCoordsButton
        self.startTrackingButton = startTrackingButton

        self.selectCoordsButton.Disable()
        self.selectDetectionAreaButton.Disable()
        self.addAdditionalAreaButton.Disable()
        self.saveAreasButton.Disable()
        self.loadCoordsButton.Disable()
        self.startTrackingButton.Disable()

        self.Maximize(True)
        self.SetTitle("Object Tracking Application")

        self.Bind(wx.EVT_SIZE, self.OnResize)

    def OnQuit(self, event):
        self.Close()

    def OnOpen(self, event):
        with wx.FileDialog(self, "Open video file", wildcard="(*.mp4;*.avi;*.mov)|*.mp4;*.avi;*.mov",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            self.video_path = fileDialog.GetPath()
            self.ResetAreas()
            self.LoadVideoFrame()
            self.selectCoordsButton.Enable()
            self.selectDetectionAreaButton.Enable()
            self.addAdditionalAreaButton.Enable()
            self.loadCoordsButton.Enable()
            self.startTrackingButton.Enable()

    def ResetAreas(self):
        """Resets the area data when opening a new video."""
        self.coordinates = []
        self.real_life_coords = []
        self.detection_area = []
        self.additional_areas = []
        self.static_bitmap = None
        self.saveAreasButton.Disable()

    def LoadVideoFrame(self):
        if not self.video_path:
            wx.MessageBox('No video path specified.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        self.cam = cv2.VideoCapture(self.video_path)
        ret, self.frame = self.cam.read()
        if not ret:
            wx.MessageBox('Unable to read video frame.', 'Error', wx.OK | wx.ICON_ERROR)
        self.cam.release()

        self.AdjustVideoSize()

    def AdjustVideoSize(self):
        if self.frame is None:
            return

        window_width, window_height = self.GetClientSize()
        height, width = self.frame.shape[:2]
        scale_factor = min((window_width - 40) / width, (window_height - 150) / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        self.scaled_frame = cv2.resize(self.frame, (new_width, new_height))

        self.scale_factor_x = width / new_width
        self.scale_factor_y = height / new_height

        self.DrawAreas()

    def OnResize(self, event):
        self.AdjustVideoSize()
        event.Skip()

    def ShowFrame(self, mode):
        if self.frame is None:
            return

        frame_copy = self.scaled_frame.copy()
        height, width = frame_copy.shape[:2]
        image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, image)

        new_window = wx.Frame(self, title=f"Select {mode.capitalize()} Area", size=(width + 40, height + 80))
        panel = wx.Panel(new_window)

        static_bitmap = wx.StaticBitmap(panel, -1, bitmap, pos=(10, 10))
        static_bitmap.Bind(wx.EVT_LEFT_DOWN, lambda event: self.OnClick(event, mode, static_bitmap, new_window))

        new_window.Show()

    def OnClick(self, event, mode, static_bitmap, window):
        x, y = event.GetPosition()
        if mode == 'coordinates':
            if len(self.coordinates) < 4:
                self.coordinates.append((int(x * self.scale_factor_x), int(y * self.scale_factor_y)))
                real_x, real_y = self.PromptForRealCoordinates(window)
                if real_x is None or real_y is None:
                    self.coordinates = []
                    self.real_life_coords = []
                    wx.MessageBox('You cancelled entering coordinates. Please enter all 4 coordinates again.', 'Error',
                                  wx.OK | wx.ICON_ERROR)
                    return
                self.real_life_coords.append((real_x, real_y))
                self.DrawOnBitmap(static_bitmap, mode)
            if len(self.coordinates) == 4:
                wx.MessageBox('Perspective transform coordinates selected.', 'Info', wx.OK | wx.ICON_INFORMATION)
                static_bitmap.Unbind(wx.EVT_LEFT_DOWN)
                window.Close()

        elif mode == 'detection':
            if len(self.detection_area) < 4:
                self.detection_area.append((int(x * self.scale_factor_x), int(y * self.scale_factor_y)))
                self.DrawOnBitmap(static_bitmap, mode)
            if len(self.detection_area) == 4:
                wx.MessageBox('Detection area selected.', 'Info', wx.OK | wx.ICON_INFORMATION)
                static_bitmap.Unbind(wx.EVT_LEFT_DOWN)
                window.Close()

        elif mode == 'additional':
            if len(self.additional_areas[-1]) < 4:
                self.additional_areas[-1].append((int(x * self.scale_factor_x), int(y * self.scale_factor_y)))
                self.DrawOnBitmap(static_bitmap, mode)
            if len(self.additional_areas[-1]) == 4:
                wx.MessageBox(f'Additional area {len(self.additional_areas)} selected.', 'Info',
                              wx.OK | wx.ICON_INFORMATION)
                static_bitmap.Unbind(wx.EVT_LEFT_DOWN)
                window.Close()

        self.AdjustVideoSize()
        self.CheckEnableSaveAreas()

    def DrawOnBitmap(self, static_bitmap, mode):
        width, height = self.scaled_frame.shape[1], self.scaled_frame.shape[0]
        image = cv2.cvtColor(self.scaled_frame, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, image)
        dc = wx.MemoryDC(bitmap)
        gc = wx.GraphicsContext.Create(dc)

        if mode == 'coordinates':
            gc.SetPen(wx.Pen(wx.Colour(0, 255, 0), 2))
            for i, coord in enumerate(self.coordinates):
                scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                if i > 0:
                    scaled_prev = (int(self.coordinates[i - 1][0] / self.scale_factor_x),
                                   int(self.coordinates[i - 1][1] / self.scale_factor_y))
                    gc.StrokeLine(*scaled_prev, *scaled_coord)
            if len(self.coordinates) == 4:
                scaled_first = (
                int(self.coordinates[0][0] / self.scale_factor_x), int(self.coordinates[0][1] / self.scale_factor_y))
                scaled_last = (
                int(self.coordinates[3][0] / self.scale_factor_x), int(self.coordinates[3][1] / self.scale_factor_y))
                gc.StrokeLine(*scaled_first, *scaled_last)

        elif mode == 'detection':
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            for i, coord in enumerate(self.detection_area):
                scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                if i > 0:
                    scaled_prev = (int(self.detection_area[i - 1][0] / self.scale_factor_x),
                                   int(self.detection_area[i - 1][1] / self.scale_factor_y))
                    gc.StrokeLine(*scaled_prev, *scaled_coord)
            if len(self.detection_area) == 4:
                scaled_first = (int(self.detection_area[0][0] / self.scale_factor_x),
                                int(self.detection_area[0][1] / self.scale_factor_y))
                scaled_last = (int(self.detection_area[3][0] / self.scale_factor_x),
                               int(self.detection_area[3][1] / self.scale_factor_y))
                gc.StrokeLine(*scaled_first, *scaled_last)

        elif mode == 'additional':
            gc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
            for i, coord in enumerate(self.additional_areas[-1]):
                scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                if i > 0:
                    scaled_prev = (int(self.additional_areas[-1][i - 1][0] / self.scale_factor_x),
                                   int(self.additional_areas[-1][i - 1][1] / self.scale_factor_y))
                    gc.StrokeLine(*scaled_prev, *scaled_coord)
            if len(self.additional_areas[-1]) == 4:
                scaled_first = (int(self.additional_areas[-1][0][0] / self.scale_factor_x),
                                int(self.additional_areas[-1][0][1] / self.scale_factor_y))
                scaled_last = (int(self.additional_areas[-1][3][0] / self.scale_factor_x),
                               int(self.additional_areas[-1][3][1] / self.scale_factor_y))
                gc.StrokeLine(*scaled_first, *scaled_last)

        dc.SelectObject(wx.NullBitmap)
        static_bitmap.SetBitmap(bitmap)
        static_bitmap.Refresh()

    def DrawAreas(self):
        if self.scaled_frame is None:
            return

        frame_copy = self.scaled_frame.copy()
        height, width = frame_copy.shape[:2]
        image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, image)
        dc = wx.MemoryDC(bitmap)
        gc = wx.GraphicsContext.Create(dc)

        # Draw perspective transform area
        if len(self.coordinates) == 4:
            gc.SetPen(wx.Pen(wx.Colour(0, 255, 0), 2))
            for i, coord in enumerate(self.coordinates):
                scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                if i > 0:
                    scaled_prev = (int(self.coordinates[i - 1][0] / self.scale_factor_x),
                                   int(self.coordinates[i - 1][1] / self.scale_factor_y))
                    gc.StrokeLine(*scaled_prev, *scaled_coord)
            scaled_first = (
            int(self.coordinates[0][0] / self.scale_factor_x), int(self.coordinates[0][1] / self.scale_factor_y))
            scaled_last = (
            int(self.coordinates[3][0] / self.scale_factor_x), int(self.coordinates[3][1] / self.scale_factor_y))
            gc.StrokeLine(*scaled_first, *scaled_last)

        # Draw detection area
        if len(self.detection_area) == 4:
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 255), 2))
            for i, coord in enumerate(self.detection_area):
                scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                if i > 0:
                    scaled_prev = (int(self.detection_area[i - 1][0] / self.scale_factor_x),
                                   int(self.detection_area[i - 1][1] / self.scale_factor_y))
                    gc.StrokeLine(*scaled_prev, *scaled_coord)
            scaled_first = (
            int(self.detection_area[0][0] / self.scale_factor_x), int(self.detection_area[0][1] / self.scale_factor_y))
            scaled_last = (
            int(self.detection_area[3][0] / self.scale_factor_x), int(self.detection_area[3][1] / self.scale_factor_y))
            gc.StrokeLine(*scaled_first, *scaled_last)

        # Draw additional areas
        gc.SetPen(wx.Pen(wx.Colour(255, 0, 0), 2))
        for area in self.additional_areas:
            if len(area) == 4:
                for i, coord in enumerate(area):
                    scaled_coord = (int(coord[0] / self.scale_factor_x), int(coord[1] / self.scale_factor_y))
                    if i > 0:
                        scaled_prev = (
                        int(area[i - 1][0] / self.scale_factor_x), int(area[i - 1][1] / self.scale_factor_y))
                        gc.StrokeLine(*scaled_prev, *scaled_coord)
                scaled_first = (int(area[0][0] / self.scale_factor_x), int(area[0][1] / self.scale_factor_y))
                scaled_last = (int(area[3][0] / self.scale_factor_x), int(area[3][1] / self.scale_factor_y))
                gc.StrokeLine(*scaled_first, *scaled_last)

        dc.SelectObject(wx.NullBitmap)
        self.bitmap = bitmap

        if self.static_bitmap is None:
            self.static_bitmap = wx.StaticBitmap(self.panel, -1, self.bitmap, pos=(10, 60))
        else:
            self.static_bitmap.SetBitmap(self.bitmap)
        self.panel.Refresh()

    def OnSelectCoords(self, event):
        if self.frame is not None:
            self.coordinates = []
            self.real_life_coords = []
            self.ShowFrame('coordinates')

    def OnSelectDetectionArea(self, event):
        if self.frame is not None:
            self.detection_area = []
            self.ShowFrame('detection')

    def OnAddAdditionalArea(self, event):
        if len(self.additional_areas) >= 6:
            wx.MessageBox('Maximum number of additional areas reached.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        if self.frame is not None:
            self.additional_areas.append([])
            self.ShowFrame('additional')

    def OnSaveAreas(self, event):
        if not self.coordinates or not self.detection_area:
            wx.MessageBox('Please select perspective transform area and detection area before saving.', 'Error',
                          wx.OK | wx.ICON_ERROR)
            return
        save_dialog = wx.FileDialog(self, "Save areas", wildcard="(*.json)|*.json",
                                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                    defaultFile=os.path.splitext(os.path.basename(self.video_path))[0] + "_areas.json")
        if save_dialog.ShowModal() == wx.ID_OK:
            self.SaveCoordinates(save_dialog.GetPath())

    def CheckEnableSaveAreas(self):
        if self.coordinates and self.detection_area:
            self.saveAreasButton.Enable()
        else:
            self.saveAreasButton.Disable()

    def OnLoadCoords(self, event):
        with wx.FileDialog(self, "Open coordinates file", wildcard="(*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path_to_coords = fileDialog.GetPath()
            self.LoadCoordinates(path_to_coords)
            self.LoadVideoFrame()
            self.CheckEnableSaveAreas()

    def LoadCoordinates(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
            self.coordinates = data.get('coordinates', [])
            self.real_life_coords = data.get('real_life_coords', [])
            self.detection_area = data.get('detection_area', [])
            self.additional_areas = data.get('additional_areas', [])
        print("Loaded coordinates:", self.coordinates)
        print("Loaded real-life coordinates:", self.real_life_coords)
        print("Loaded detection area:", self.detection_area)
        print("Loaded additional areas:", self.additional_areas)

    def SaveCoordinates(self, path):
        data = {
            'coordinates': self.coordinates,
            'real_life_coords': self.real_life_coords,
            'detection_area': self.detection_area,
            'additional_areas': self.additional_areas
        }
        with open(path, 'w') as file:
            json.dump(data, file)
        print("Coordinates saved to", path)

    def PromptForRealCoordinates(self, window):
        dlg = wx.TextEntryDialog(window, "Enter the real-life X coordinate (default=0):", "Real-life X Coordinate", "0")
        if dlg.ShowModal() == wx.ID_OK:
            real_x = dlg.GetValue()
        else:
            dlg.Destroy()
            return None, None
        dlg.Destroy()

        dlg = wx.TextEntryDialog(window, "Enter the real-life Y coordinate (default=0):", "Real-life Y Coordinate", "0")
        if dlg.ShowModal() == wx.ID_OK:
            real_y = dlg.GetValue()
        else:
            dlg.Destroy()
            return None, None
        dlg.Destroy()

        return float(real_x), float(real_y)

    def OnStartTracking(self, event):
        if not self.video_path or not self.coordinates or not self.real_life_coords or not self.detection_area:
            wx.MessageBox('Please select video, perspective transform area, and detection area.', 'Error',
                          wx.OK | wx.ICON_ERROR)
            return

        self.img_size = int(self.imgSizeChoice.GetStringSelection())
        self.confidence = float(self.confChoice.GetStringSelection())

        start_tracking(self.coordinates, self.real_life_coords, self.video_path, self.detection_area,
                       self.additional_areas, self.img_size, self.confidence)
        self.EnableButtons(True)

    def EnableButtons(self, state):
        self.selectCoordsButton.Enable(state)
        self.selectDetectionAreaButton.Enable(state)
        self.addAdditionalAreaButton.Enable(state)
        self.saveAreasButton.Enable(state)
        self.loadCoordsButton.Enable(state)
        self.startTrackingButton.Enable(state)
        self.selectCoordsButton.Refresh()
        self.selectDetectionAreaButton.Refresh()
        self.addAdditionalAreaButton.Refresh()
        self.saveAreasButton.Refresh()
        self.loadCoordsButton.Refresh()
        self.startTrackingButton.Refresh()


def main():
    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
