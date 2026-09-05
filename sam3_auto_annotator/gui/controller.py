"""Compatibility imports for the retired monolithic GUI controller.

New code imports from ``sam3_auto_annotator.gui.controllers``.  ``AppController``
remains as a temporary import alias so downstream code does not carry a duplicate
legacy implementation during migration.
"""

from sam3_auto_annotator.gui.controllers.state import UiMode
from sam3_auto_annotator.gui.controllers.workstation_controller import WorkstationController

AppController = WorkstationController

__all__ = ["AppController", "UiMode"]
