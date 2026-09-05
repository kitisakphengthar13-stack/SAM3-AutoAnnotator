class ControllerSurfaceAdapter:
    """Temporary adapter for legacy controller surface-selection calls.

    New UI code must call MainWindow.show_setup/show_review/show_results directly.
    This adapter exists only until AppController stops referring to the retired
    inspector abstraction.
    """

    def __init__(self, window):
        self.window = window
        self._current = window.setup

    def setCurrentWidget(self, widget):
        self._current = widget
        if widget is self.window.setup:
            self.window.show_setup()
        elif widget is self.window.annotation:
            self.window.show_review()
        elif widget is self.window.results:
            self.window.show_results()

    def currentWidget(self):
        return self._current
