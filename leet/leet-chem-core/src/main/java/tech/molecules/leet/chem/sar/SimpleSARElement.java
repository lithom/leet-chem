package tech.molecules.leet.chem.sar;


import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;

/**
 *
 * 1. supports sytnhon logic, i.e. you can use connector atoms (n>=92) for assembling
 *    the scaffold.
 *
 * 2. decomposition is done purely via labeling, labeled atoms give rise to a
 *    part of the molecule
 *
 *
 * Algo for matching labeled parts: very simple,
 *     just cut off all bonds that are:
 *     1. both atoms are part of the sar scaffold
 *     2. one atom is labeled, the other one is either unlabeled or differently labeled
 *
 * Possible extensions: on the MultiFragment Level, maybe think about subdividing?
 *
 */
public class SimpleSARElement {

    private StereoMolecule mol;

    public SimpleSARElement(StereoMolecule mol) {
        this.mol = mol;
        this.mol.ensureHelperArrays(Molecule.cHelperCIP);
    }

    public StereoMolecule getMol() {
        return this.mol;
    }

    public String getAtomLabel(int a) {
        String sa = mol.getAtomCustomLabel(a);
        if(sa!=null && !sa.isEmpty()) {
            return sa;
        }
        return null;
    }

}
