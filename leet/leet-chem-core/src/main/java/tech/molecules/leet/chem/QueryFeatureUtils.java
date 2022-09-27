package tech.molecules.leet.chem;

import com.actelion.research.chem.ExtendedMolecule;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;

public class QueryFeatureUtils {

    /**
     *
     *
     * Removes the following restricting query features:
     *
     * Atoms:
     *
     *
     * @param mol
     */
    public static void removeNarrowingQueryFeatures(ExtendedMolecule mol) {

        if(!mol.isFragment()) {
            return;
        }

        for (int atom=0; atom<mol.getAllAtoms(); atom++) {
            mol.setAtomQueryFeature(atom, Molecule.cAtomQFNarrowing, false);
            mol.setAtomQueryFeature(atom, Molecule.cAtomQFNoMoreNeighbours,false);
        }
        for (int bond=0; bond<mol.getAllBonds(); bond++) {
            mol.setBondQueryFeature(bond, Molecule.cBondQFNarrowing, false);
        }

    }

    public static void addBridgeBond(StereoMolecule mi, int atom_a, int atom_b, int minlength, int maxlength) {
        int bi = mi.addBond(atom_a,atom_b);
        setBridgeBondQF(mi,bi,minlength,maxlength);
    }

    public static void setBridgeBondQF(StereoMolecule mi, int bond, int minlength, int maxlength) {
        int minAtoms = minlength;
        int atomSpan = maxlength-minlength;

        int queryFeatures = 0;
        queryFeatures |= (minAtoms << Molecule.cBondQFBridgeMinShift);
        queryFeatures |= (atomSpan << Molecule.cBondQFBridgeSpanShift);
        queryFeatures &= ~Molecule.cBondQFBondTypes;

        mi.setBondQueryFeature(bond,queryFeatures,true);
    }

}
