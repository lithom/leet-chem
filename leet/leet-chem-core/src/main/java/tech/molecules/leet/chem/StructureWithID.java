package tech.molecules.leet.chem;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.reflect.ConstructorUtils;

import java.io.Serializable;

public class StructureWithID implements Serializable {
    public final String structure[];
    public final String molid;
    public final String batchid;

    public StructureWithID(String molid, String batchid, String[] struc) {
        this.structure = struc;
        this.molid = molid;
        this.batchid = batchid;
    }

    public StructureWithID(String molid, String batchid, StereoMolecule mol) {
        this.structure = new String[]{ mol.getIDCode() , mol.getIDCoordinates() };
        this.molid = molid;
        this.batchid = batchid;
    }

    public StructureWithID(String idc) {
        this(idc,"",ChemUtils.parseIDCode(idc));
    }
    public StructureWithID(StereoMolecule mi) {
        this(mi.getIDCode(),"",mi);
    }

}