package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.chem.util.PoolManager;
import tech.molecules.leet.datatable.AbstractCachedDataFilter;
import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.filter.NumericRangeFilter;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

public class SubstructureFilter<T extends StructureRecord> extends AbstractCachedDataFilter<T> {

    // determines the max number of threads for parallel filtering
    private int numSearchers = 8;

    private StereoMolecule substructure = null;
    private BitSet substructureFFP = null;
    //private SSSearcher ssi = new SSSearcher

    private PoolManager<SSSearcher> searcherPool;

    public void setSubstructure(StereoMolecule ss) {
        this.substructure = ss;
        this.substructureFFP = BitSet.valueOf(DescriptorHandlerLongFFP512.getDefaultInstance().createDescriptor(ss));

        List<SSSearcher> searchers = new ArrayList<>();
        for(int zi=0;zi<numSearchers;zi++) {
            SSSearcher ssi = new SSSearcher();
            ssi.setFragment(ss);
            searchers.add(ssi);
        }
        this.searcherPool = new PoolManager<>(searchers);
        fireFilterChanged();
    }


    /**
     *
     *
     * @param vi
     * @return false if substructure is contained, true otherwise
     */
    @Override
    public boolean filterRow(StructureRecord vi) {
        if(this.substructure!=null) {
            // ffp check:
            BitSet bsi = (BitSet) vi.ffp.clone();
            bsi.or(this.substructureFFP);
            if(bsi.cardinality() > vi.ffp.cardinality()) {
                return true;
            }
            boolean match = false;
            try {
                // perform substructure matching:
                SSSearcher ssi = this.searcherPool.acquireSearcher();
                ssi.setMolecule(ChemUtils.parseIDCode(vi.structure[0],vi.structure[1]));
                match = ssi.isFragmentInMolecule();
                this.searcherPool.releaseSearcher(ssi);
            } catch (InterruptedException e) {
                System.out.println("[ERROR] InterruptedException during substructure filtering..");
                return false;
            }

            return match;
        }
        return false;
    }

    @Override
    public DataFilterType<T> getDataFilterType() {
        return new SubstructureFilterType<>();
    }

    @Override
    public double getApproximateFilterSpeed() {
        return 0.1;
    }


    public static class SubstructureFilterType<T extends StructureRecord> implements DataFilterType<T> {
        @Override
        public String getFilterName() {
            return "SubstructureFilterType";
        }

        @Override
        public boolean requiresInitialization() {
            return true;
        }

        @Override
        public DataFilter<T> createInstance(DataTableColumn column) {
            return new SubstructureFilter();
        }
    }

}
