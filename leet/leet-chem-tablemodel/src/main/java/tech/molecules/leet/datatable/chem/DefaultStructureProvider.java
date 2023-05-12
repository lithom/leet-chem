package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.dataprovider.HashMapBasedDataProvider;

import java.util.*;

public class DefaultStructureProvider extends HashMapBasedDataProvider<StructureWithID> {

    //private Map<String, StructureWithID> data;

    public DefaultStructureProvider() {
        super(new HashMap<>());
        //this.data = new HashMap<>();
    }

    public void loadStructures(Collection<StructureWithID> structures) {
        Map<String,StructureWithID> newData = new HashMap<>();
        List<String> changed_keys = new ArrayList<>();
        for(StructureWithID sid : structures) {
            if(sid.batchid!=null && !sid.batchid.isEmpty()) {
                newData.put(sid.batchid,sid);
                changed_keys.add(sid.batchid);
            }
            if(sid.molid!=null && !sid.molid.isEmpty()) {
                newData.put(sid.molid,sid);
                changed_keys.add(sid.molid);
            }
            newData.put(sid.structure[0],sid);
            changed_keys.add(sid.structure[0]);
        }
        this.addData(newData);
    }

    public void loadStructures(Collection<Pair<String,String[]>> structures, boolean setIdInSID) {
        Map<String,StructureWithID> newData = new HashMap<>();
        List<String> changed_keys = new ArrayList<>();
        for(Pair<String,String[]> sid : structures) {
            if(setIdInSID) {
                newData.put(sid.getLeft(),new StructureWithID(sid.getLeft(),"",sid.getRight()) );
                changed_keys.add(sid.getLeft());
            }
            else {
                newData.put(sid.getLeft(),new StructureWithID("","",sid.getRight()) );
                changed_keys.add(sid.getLeft());
            }
        }
        this.addData(newData);
    }

}
