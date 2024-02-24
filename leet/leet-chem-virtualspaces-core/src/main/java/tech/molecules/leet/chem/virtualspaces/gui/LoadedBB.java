package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;

public class LoadedBB {

    private String molid;
    private String idcode;
    private int numAtoms;

    public LoadedBB(String molid, String idcode, StereoMolecule mi) {
        this.molid = molid;
        this.idcode = idcode;
        mi.ensureHelperArrays(Molecule.cHelperNeighbours);
        this.numAtoms = mi.getAtoms();
    }

    public String getIdcode() {
        return idcode;
    }

    public int getNumAtoms() {
        return numAtoms;
    }

    public String getMolid() {
        return molid;
    }
}
