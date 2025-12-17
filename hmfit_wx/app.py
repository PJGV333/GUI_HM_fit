from __future__ import annotations


def main() -> int:
    try:
        import wx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "wxPython no está instalado. Instala wxPython y reintenta."
        ) from exc

    from hmfit_wx.panels.nmr import NMRPanel
    from hmfit_wx.panels.spectroscopy import SpectroscopyPanel

    class HMFitFrame(wx.Frame):
        def __init__(self) -> None:
            super().__init__(None, title="HM Fit (wxPython)", size=(1200, 800))

            nb = wx.Notebook(self)
            nb.AddPage(SpectroscopyPanel(nb), "Spectroscopy")
            nb.AddPage(NMRPanel(nb), "NMR")

            self.Centre()

    app = wx.App(False)
    frame = HMFitFrame()
    frame.Show(True)
    app.MainLoop()
    return 0

