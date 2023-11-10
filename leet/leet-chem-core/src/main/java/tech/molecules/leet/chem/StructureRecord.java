package tech.molecules.leet.chem;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;

import java.util.BitSet;

/**
 * A StructureWithID object with an attached BitSet containing the
 * OpenChemLib FFP descriptor that can be used for substructure  filtering.
 */
public class StructureRecord extends StructureWithID {
    //public final String structure[];
    //public final String molid;
    //public final String batchid;
    public final BitSet ffp;

    public StructureRecord(String molid, String batchid, String[] struc, BitSet ffp) {
        super(molid,batchid,struc);
        //this.structure = struc;
        //this.molid = molid;
        //this.batchid = batchid;
        this.ffp = ffp;
    }

    public StructureRecord(String molid, String batchid, String[] struc) {
        super(molid,batchid,struc);
        //this.structure = struc;
        //this.molid = molid;
        //this.batchid = batchid;
        long[] ffp_a = DescriptorHandlerLongFFP512.getDefaultInstance().createDescriptor(ChemUtils.parseIDCode(struc[0],struc[1]));
        this.ffp = BitSet.valueOf(ffp_a);
    }

    public StructureRecord(String molid, String batchid, StereoMolecule mol) {
        super(molid,batchid,new String[]{ mol.getIDCode() , mol.getIDCoordinates() });
        //this.structure = new String[]{ mol.getIDCode() , mol.getIDCoordinates() };
        //this.molid = molid;
        //this.batchid = batchid;
        long[] ffp_a = DescriptorHandlerLongFFP512.getDefaultInstance().createDescriptor(mol);
        this.ffp = BitSet.valueOf(ffp_a);
    }

    public StructureRecord(String idc) {
        this(idc,"",ChemUtils.parseIDCode(idc));
    }
    public StructureRecord(StereoMolecule mi) {
        this(mi.getIDCode(),"",mi);
    }
}
