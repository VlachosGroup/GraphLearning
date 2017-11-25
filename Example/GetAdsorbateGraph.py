
import os
from GraphLearning.io import adsorbate
from GraphLearning.GraphMining import FindAllAdsorbateGraphsOfLengthN
################################ USER INPUT ###################################
InputPath = '.\\Input\\'
SurfaceAtomSymbols = ['Pt']
################################ USER INPUT ###################################
# Get SMILES
for datum in os.listdir(InputPath):
    fpath = InputPath + datum + '\\CONTCAR'
    # Convert atomic coordinates (VASP) to molecular graph.
    mole = adsorbate.LoadByCovalentRadius(fpath, SurfaceAtomSymbols = SurfaceAtomSymbols)
    # Mine adsorbate graphs and print
    print FindAllAdsorbateGraphsOfLengthN(mole.RdkitMol,SurfaceAtomSymbols,1)
