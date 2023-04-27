package tech.molecules.leet.datatable;

import java.util.BitSet;
import java.util.List;

public interface DataFilter<T> {

    public abstract String getFilterName();

    /**
     *
     * @param data
     * @param ids
     * @param filtered bits that are one indicate rows that are not yet filtered. For these
     *                 the function has to check if it should be filtered, and in that case
     *                 for the given position in the result bitset a zero must be returned.
     * @return
     */
    public BitSet filterRows(DataProvider<T> data, List<String> ids, BitSet filtered);
    public double getApproximateFilterSpeed();
    public void setupFilter(DataProvider<T> dp, List<String> ids);

    /**
     * For certain filters it may be possible that the column first has
     * to initialize specific datastructures asynchronously. In this case,
     * the filter will return false in this function until the data is ready.
     * Example would be the substructure filter and a structure column (loading
     * fingerprints asynchronously).
     *
     * @return
     */
    public FilterState getFilterState();

    public static enum FilterState {READY,UPDATING};

    public void addFilterListener(FilterListener li);

    public boolean removeFilterListener(FilterListener li);

    public static interface FilterListener {
        public void filterStateChanged(FilterState si);
    }

}