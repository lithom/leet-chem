package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.dataprovider.DataProviderListenerHelper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StructurePropertiesProvider implements DataProvider<StructurePropertiesProvider.CalculatedBasicStructureProperties> {

    private DataProviderListenerHelper listenerHelper = new DataProviderListenerHelper();
    private DataProvider<StructureWithID> structureDataProvider;

    public StructurePropertiesProvider(DataProvider<StructureWithID> dp) {
        this.structureDataProvider = dp;
        dp.addDataProviderListener(new DataProviderListener() {
            @Override
            public void dataChanged(List<String> keysChanged) {
                listenerHelper.fireDataChanged(keysChanged);
            }
        });
    }

    public static class CalculatedBasicStructureProperties {
        /**
         * indeces are positions in ChemPropertyCounts.COUNTS_ALL
         */
        public Map<Integer,Integer> counts = new HashMap<>();

        public CalculatedBasicStructureProperties(Map<Integer,Integer> counts) {
            this.counts = new HashMap<>(counts);
        }
    }


    @Override
    public List<String> getAllEntries() {
        return structureDataProvider.getAllEntries();
    }

    @Override
    public CalculatedBasicStructureProperties getData(String key) {
        StructureWithID sid = structureDataProvider.getData(key);
        if(sid==null || sid.structure==null) {
            return null;
        }

        StereoMolecule m = new StereoMolecule();
        IDCodeParser icp = new IDCodeParser();
        icp.parse(m,sid.structure[0],sid.structure[1]);
        m.ensureHelperArrays(Molecule.cHelperCIP);

        Map<Integer,Integer> counts = new HashMap<>();
        for(int zi = 0; zi< ChemPropertyCounts.COUNTS_ALL.length; zi++) {
            counts.put( zi, ChemPropertyCounts.COUNTS_ALL[zi].evaluator.apply(m) );
        }
        return new CalculatedBasicStructureProperties(counts);
    }

    @Override
    public void addDataProviderListener(DataProviderListener li) {
        this.listenerHelper.addDataProviderListener(li);
    }

    @Override
    public boolean removeDataProviderListener(DataProviderListener li) {
        return this.listenerHelper.removeDataProviderListener(li);
    }
}
