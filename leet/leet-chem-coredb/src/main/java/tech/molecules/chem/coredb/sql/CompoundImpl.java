package tech.molecules.chem.coredb.sql;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.Compound;
import tech.molecules.leet.chem.ChemUtils;

public class CompoundImpl implements Compound {
    private String id;
    private String idcode;
    private String idcode_coordinates;

    public CompoundImpl(String id, StereoMolecule molecule) {
        this.id = id;
        this.idcode = molecule.getIDCode();
        this.idcode_coordinates = molecule.getIDCoordinates();
    }

    public CompoundImpl(String id, String idcode) {
        this.id = id;
        this.idcode = idcode;
    }


    public String getId() { return id; }
    public String getIdcode() { return idcode; }
    public String[] getMolecule() { return new String[]{idcode,idcode_coordinates}; }
}