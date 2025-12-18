from __future__ import annotations


def main() -> int:
    import sys

    import wx

    from hmfit_wx_legacy.GUI_interface_wxpy import App

    if sys.platform.startswith("win"):
        # Enable High-DPI awareness on Windows (best effort).
        import os

        os.environ.setdefault("WX_HIGH_DPI_AWARE", "1")

    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

