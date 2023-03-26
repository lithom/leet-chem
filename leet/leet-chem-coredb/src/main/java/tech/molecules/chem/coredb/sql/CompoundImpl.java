package tech.molecules.chem.coredb.sql;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.Compound;
import tech.molecules.leet.chem.ChemUtils;

public class CompoundImpl implements Compound {
    private String id;
    private String idcode;

    public CompoundImpl(String id, StereoMolecule molecule) {
        this.id = id;
        this.idcode = molecule.getIDCode();
    }

    public CompoundImpl(String id, String idcode) {
        this.id = id;
        this.idcode = idcode;
    }


    public String getId() { return id; }
    public String getIdcode() { return idcode; }
    public StereoMolecule getMolecule() { return ChemUtils.parseIDCode(idcode); }
}