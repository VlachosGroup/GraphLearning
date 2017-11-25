"""
Geun Ho Gu, Vlachos Group, University of Delaware, 2017
"""
from ase.io import read
import numpy as np
from rdkit import Chem
from ase import Atoms as ase_Atoms

class adsorbate(object):
    """
    This is an adsorbate graph class that converts atomic coordinates to rdkit 
    molecular graph object, Mol. Use "LoadByCovalentRadius" to initialize.
    
    Class Variables
    soan: selected organic atomic number. These atoms are considered adosbates
    rcov: covalent radius. Info available in wikipedia.
    
    Class Attributes
    ASEAtoms:                   ASE Atoms object.
    RdkitMol:                   Rdkit Mol object.
    SurfaceAtomSymbols:         List of symbols of surface atoms.
    ASEAtomIndex2RdKitAtomIndex: Index mapping from ASE atoms to Rdkit Mol
    RdKitAtomIndex2ASEAtomIndex: Index mapping from Rdkit Mol to ASE Atoms.
    """
    # selected organic atomic numbers
    soan = [1,6,7,8] 
    # atomic number -> covalent radius
    rcov = {1:0.31, 6:0.76, 7:0.71, 8:0.66, 26:1.26, 27:1.21, 28:1.21, 29:1.21,\
            44:1.16, 45:1.21, 46:1.26, 47:1.46, 75:1.21, 77:1.21 ,78:1.21, 79:1.21}
    
    def __init__(self,ASEAtoms,RdkitMol,SurfaceAtomSymbols, \
                 ASEAtomIndex2RdKitAtomIndex, RdKitAtomIndex2ASEAtomIndex):
        
        assert isinstance(ASEAtoms,ase_Atoms)
        assert isinstance(RdkitMol,Chem.Mol)
        assert isinstance(ASEAtomIndex2RdKitAtomIndex,dict)
        assert isinstance(RdKitAtomIndex2ASEAtomIndex,dict)
        if isinstance(SurfaceAtomSymbols,str):
            SurfaceAtomSymbols = [SurfaceAtomSymbols]
        else:
            assert isinstance(SurfaceAtomSymbols,list)
        self.ASEAtoms = ASEAtoms
        self.RdkitMol = RdkitMol
        self.SurfaceAtomSymbols = SurfaceAtomSymbols
        self.ASEAtomIndex2RdKitAtomIndex = ASEAtomIndex2RdKitAtomIndex
        self.RdKitAtomIndex2ASEAtomIndex = RdKitAtomIndex2ASEAtomIndex
    
    @classmethod
    def LoadByCovalentRadius(cls,CoordinateFPath, SurfaceAtomSymbols, \
        rfacup = 1.35,rfacdown = 0.6, z_vector = 2):
        """ 
        This function reads file using ASE read, and construts molecular graph
        in rdkit object, Mol. See manuscript for overall algorithm.
        
        
        Input List
        CoordinateFPath:    path to ASE readable coordinate file.
        SurfaceAtomSymbols: List of atomic symbols of surface atoms.
        rfacup:             Upper percentage limit for determining connectivity.
        rfacdown:           Lower percentage limit for determining connectivity.
        z_vector:           index of cell basis vector that is orthogonal to surface.
        
        Output List
        adsorbate class
        """
        
        # initialize
        ASEAtomIndex2RdKitAtomIndex = dict()
        RdKitAtomIndex2ASEAtomIndex = dict()
        if isinstance(SurfaceAtomSymbols,str):
            SurfaceAtomSymbols = [SurfaceAtomSymbols]
        else:
            assert isinstance(SurfaceAtomSymbols,list)
        # load POSCAR
        AseAtoms = read(CoordinateFPath)
        # if none given for surface layer z coordinate, average the top layer atomic coordinate
        _, SurfaceAtomIndex = DetermineSurfaceLayerZ(AseAtoms, SurfaceAtomSymbols, ZVecIndex = z_vector)

        # (p)eriodic (b)oundary (c)ondition(s)
        PBCs = [[0,0,0]]
        if AseAtoms.pbc[0]:
            temp = np.add(PBCs,[1,0,0])
            temp = np.concatenate((temp,np.add(PBCs,[-1,0,0])))
            PBCs = np.concatenate((PBCs,temp))
        if AseAtoms.pbc[1]:
            temp = np.add(PBCs,[0,1,0])
            temp = np.concatenate((temp,np.add(PBCs,[0,-1,0])))
            PBCs = np.concatenate((PBCs,temp))
        if AseAtoms.pbc[2]:
            temp = np.add(PBCs,[0,0,1])
            temp = np.concatenate((temp,np.add(PBCs,[0,0,-1])))
            PBCs = np.concatenate((PBCs,temp))
                    
        # Get organic atoms from the DFT calculations (their index and atomic number)
        ans = AseAtoms.get_atomic_numbers() # (a)tomic (n)umber(s)
        oai = list() #organic atom index in the atoms object
        oan = list() #organic atomic number
        for i in xrange(0,AseAtoms.__len__()):
            if ans[i] in cls.soan:
                oai.append(i)
                oan.append(ans[i])
        
        # Determine connectivity of the organic atoms
        adj_mat = np.zeros((oai.__len__(),oai.__len__())) # adjacency matrix
        for i in xrange(0,oai.__len__()):
            for j in xrange(i+1,oai.__len__()):
                if cls._DetermineConnectivity(AseAtoms,oai[i],oai[j],PBCs,rfacup,rfacdown):
                    adj_mat[i,j] = 1
        
        # construct mol object
        RdkitMol = Chem.Mol()
        RdkitMol = Chem.RWMol(RdkitMol)
        
        ## add atom
        ### organic atoms
        for i in xrange(0,oan.__len__()):
            atom = Chem.Atom(oan[i])
            atom.SetNoImplicit(True) # this allows molecule to have radical atoms
            atom.SetBoolProp('Adsorbed',False)
            RdkitMol.AddAtom(atom)
            ASEAtomIndex2RdKitAtomIndex[oai[i]] = i
            RdKitAtomIndex2ASEAtomIndex[i] = oai[i]
        ### surface atoms
        for index in SurfaceAtomIndex:
            atom = Chem.Atom(AseAtoms[index].symbol)
            atom.SetBoolProp('SurfaceAtom',True)
            atom.SetBoolProp('Occupied',False)
            i = RdkitMol.AddAtom(atom)
            ASEAtomIndex2RdKitAtomIndex[index] = i
            RdKitAtomIndex2ASEAtomIndex[i] = index
        
        ## add bond
        ### between organic atoms
        for i in xrange(0,oai.__len__()):
            for j in xrange(i+1,oai.__len__()):
                if adj_mat[i,j] == 1:
                    RdkitMol.AddBond(i,j,order=Chem.rdchem.BondType.SINGLE)
                    
        ### between surface atoms
        for i in xrange(0,len(SurfaceAtomIndex)):
            for j in xrange(i+1,len(SurfaceAtomIndex)):
                if cls._DetermineConnectivity(AseAtoms,SurfaceAtomIndex[i],SurfaceAtomIndex[j],PBCs,rfacup,rfacdown):
                    RdkitMol.AddBond(ASEAtomIndex2RdKitAtomIndex[SurfaceAtomIndex[i]],ASEAtomIndex2RdKitAtomIndex[SurfaceAtomIndex[j]],order=Chem.rdchem.BondType.ZERO)
                    
        ## assign radicals
        Chem.AssignRadicals(RdkitMol)
        
        ## set smilesSymbol
        for atom in RdkitMol.GetAtoms():
            if atom.GetSymbol() in ['C','O'] and atom.GetNumRadicalElectrons() == 0:
                atom.SetProp("smilesSymbol",'[' + atom.GetSymbol() + str(atom.GetNumRadicalElectrons())+ ']')
            elif atom.GetNumRadicalElectrons() > 0:
                atom.SetProp("smilesSymbol",atom.GetSymbol() + str(atom.GetNumRadicalElectrons()))
            
        # Find surface binding atom. This is done by finding all the radical atoms
        rai_rdkit = list() # radical atom index for rdkit mol
        rai_ase = list() # radical atom index for rdkit ase atoms object
        for atom in RdkitMol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                rai_rdkit.append(atom.GetIdx())
                rai_ase.append(oai[atom.GetIdx()])
        
        # Surface connectivity
        for i in xrange(0,len(rai_ase)):
            for j in xrange(0,len(SurfaceAtomIndex)):
                if cls._DetermineConnectivity(AseAtoms,rai_ase[i],SurfaceAtomIndex[j],PBCs,rfacup,rfacdown):
                    RdkitMol.AddBond(rai_rdkit[i],ASEAtomIndex2RdKitAtomIndex[SurfaceAtomIndex[j]],order=Chem.rdchem.BondType.ZERO)
                    RdkitMol.GetAtomWithIdx(ASEAtomIndex2RdKitAtomIndex[SurfaceAtomIndex[j]]).SetBoolProp('Occupied',True)
                    RdkitMol.GetAtomWithIdx(rai_rdkit[i]).SetBoolProp('Adsorbed',True)
        
        # assign binding site.
        for i in xrange(0,len(rai_rdkit)):
            a = RdkitMol.GetAtomWithIdx(rai_rdkit[i])
            nsurf = 0
            for neighbor_atom in a.GetNeighbors():
                if neighbor_atom.GetSymbol() in SurfaceAtomSymbols:
                    nsurf += 1
            a.SetProp("smilesSymbol",a.GetProp("smilesSymbol") + '_' + str(nsurf) + 'fold')
            
        adsorbate = cls(AseAtoms,RdkitMol,SurfaceAtomSymbols, \
                 ASEAtomIndex2RdKitAtomIndex, RdKitAtomIndex2ASEAtomIndex)
        
        return adsorbate 

    @classmethod
    def _DetermineConnectivity(cls,AseAtoms,i,j,PBCs,rfacup,rfacdown):
        """
        Determine connectivity between atom i and j. See equation (1) in the 
        manuscript.
        
        Input List
        ASEAtoms:           ASE atoms containing adsorbate/surface system
        PBCs:               Periodic Boundary Conditions. e.g., (1,0,0) means 
                            cell repeats in first basis vector but not others.
        rfacup:             upper tolerance factor
        rfacdown:           lower tolerance factor
        
        Output List
        Bool:               True if connected, false if not.
        """
        xyz1 = AseAtoms[i].position
        # loop over periodic cells
        for PBC in PBCs:
            xyz2 = AseAtoms[j].position + np.dot(PBC,AseAtoms.cell)
            # Criteria:
            # TolFaclower * ideal_distance < distance < TolFacupper * ideal_distance 
            # ideal ideal_distance = Rcov(Atom1) + Rcov(Atom2)
            d = np.linalg.norm(xyz1-xyz2) # distance
            i_d = cls.rcov[AseAtoms[i].number] + cls.rcov[AseAtoms[j].number] # ideal distance
            if d <= i_d*rfacup and d >= i_d*rfacdown:
                return True
        return False
    
    
def DetermineSurfaceLayerZ(ASEAtoms, SurfaceAtomSymbols, ZVecIndex = 2, ztol = 0.5):
    """
    Find top layer surface atom z coordinates by averaging
    atoms within ztol (angstrom) of the top most atoms are selected for averaging
    
    Input List
    ASEAtoms:           ASE atoms containing adsorbate/surface system.
    SurfaceAtomSymbols: Symbol of surface atoms.
    ZVecIndex:          index of cell basis vector that is orthogonal to surface.
    ztol:               Atoms within ztol(angstrom) of the top most atoms are selected as 
                        surface atoms.
    Output List
    SurfaceLayerZ:      z coordinate of surface layer.
    SurfaceAtomIndex:   Index of surface atoms.
    """
    assert isinstance(ASEAtoms,ase_Atoms)
    # get highest surface atom coordinate
    zmax = 0
    zs = ASEAtoms.get_scaled_positions()[:,2]
    for i in xrange(0,len(ASEAtoms)):
        if ASEAtoms[i].symbol in SurfaceAtomSymbols and zmax < zs[i]:
            zmax = zs[i]
            
    # determine z coordinate. average out top layer
    ztol = ztol/np.linalg.norm(ASEAtoms.cell[2,:])
    SurfaceAtomIndex = list()
    SurfZs = list()
    for i in xrange(0,len(ASEAtoms)):
        if ASEAtoms[i].symbol in SurfaceAtomSymbols and zmax - ztol < zs[i]:
            SurfZs.append(zs[i])
            SurfaceAtomIndex.append(i)
    SurfaceLayerZ = np.array(SurfZs).mean()

    return SurfaceLayerZ, SurfaceAtomIndex