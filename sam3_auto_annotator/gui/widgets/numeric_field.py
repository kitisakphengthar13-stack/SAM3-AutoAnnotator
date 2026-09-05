from PySide6.QtCore import QLocale
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QLineEdit


def configure_c_locale():
    QLocale.setDefault(QLocale.c())


class NumericLineEdit(QLineEdit):
    def __init__(self, value=0.0, decimals=2, minimum=0.0, maximum=999999.0, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.minimum = minimum
        self.maximum = maximum
        validator = QDoubleValidator(minimum, maximum, decimals, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setLocale(QLocale.c())
        self.setValidator(validator)
        self.set_value(value)

    def value(self):
        text = self.text().strip()
        if not text:
            raise ValueError("Numeric field is empty.")
        value, parsed = QLocale.c().toDouble(text)
        if not parsed:
            raise ValueError(f"'{text}' is not a valid decimal number.")
        if value < self.minimum or value > self.maximum:
            raise ValueError(
                f"Value {value:.{self.decimals}f} must be between "
                f"{self.minimum:.{self.decimals}f} and {self.maximum:.{self.decimals}f}."
            )
        return value

    def set_value(self, value):
        self.setText(QLocale.c().toString(float(value), "f", self.decimals))
