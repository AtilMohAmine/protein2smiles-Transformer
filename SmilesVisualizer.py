from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMainWindow
from PIL.ImageQt import ImageQt
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

class SmilesVisualizer(QMainWindow):

    def __init__(self, smiles):
        super().__init__()

        # Create a RDKit molecule object from the SMILES string
        self.mol = Chem.MolFromSmiles(smiles)

        # Create the UI elements
        self.label = QLabel()
        self.setCentralWidget(self.label)

        # Set the window title
        self.setWindowTitle("SMILES Visualizer")

        # Display the molecule image
        if self.mol is not None:
            self.display_molecule()
        else:
            print('The predicted SMILES does not correspond to a valid molecule')

    def display_molecule(self):
        # Generate an image of the molecule using RDKit
        image = MolToImage(self.mol)

        # Convert the image to a QImage for display in PyQt5
        qimage = ImageQt(image.convert('RGBA'))
        pixmap = QPixmap.fromImage(qimage)

        # Display the QPixmap in the label widget
        self.label.setPixmap(pixmap)
