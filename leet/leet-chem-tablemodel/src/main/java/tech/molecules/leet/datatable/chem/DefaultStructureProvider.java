package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataProvider;

import java.util.*;

public class DefaultStructureProvider implements DataProvider<StructureWithID> {

    private Map<String, StructureWithID> data;

    public DefaultStructureProvider() {
        this.data = new HashMap<>();
    }

    public void loadStructures(Collection<StructureWithID> structures) {
        List<String> changed_keys = new ArrayList<>();
        for(StructureWithID sid : structures) {
            if(sid.batchid!=null && !sid.batchid.isEmpty()) {
                this.data.put(sid.batchid,sid);
                changed_keys.add(sid.batchid);
            }
            if(sid.molid!=null && !sid.molid.isEmpty()) {
                this.data.put(sid.molid,sid);
                changed_keys.add(sid.molid);
            }
            this.data.put(sid.structure[0],sid);
            changed_keys.add(sid.structure[0]);
        }
        fireDataChanged( changed_keys );
    }

    @Override
    public List<String> getAllEntries() {
        return new ArrayList<>(this.data.keySet());
    }

    @Override
    public StructureWithID getData(String key) {
        return this.data.get(key);
    }

    private List<DataProviderListener> listeners = new ArrayList<>();

    private void fireDataChanged(List<String> changed_keys) {
        for(DataProviderListener li : listeners) {
            li.dataChanged(changed_keys);
        }
    }

    @Override
    public void addDataProviderListener(DataProviderListener li) {
        this.listeners.add(li);
    }

    @Override
    public boolean removeDataProviderListener(DataProviderListener li) {
        return this.listeners.remove(li);
    }
}
