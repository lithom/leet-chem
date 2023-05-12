package tech.molecules.leet.datatable;

import java.util.BitSet;
import java.util.List;

/**
 * Generally works as follows: in case that via the UI the configuration of the filter is changed
 * the filterConfigurationChanged() function of the listeners is called.
 *
 * The DataTable listens to the DataFilter, and in case of a filterConfigurationChanged event
 * an update of the filtering is triggered within the DataTable.
 *
 * @param <T>
 */
public interface DataFilter<U> {

    public enum FilterState {READY , UPDATING};

    public DataFilterType<U> getDataFilterType();

    /**
     * This method is expected to be blocking. It can spawn its own threads, but it
     * still should return only after all threads are finished.
     *
     * This method is called asynchronouslly, i.e. it is run inside a thread spawned by the
     * DataTable.
     *
     *
     * @param data
     * @param ids
     * @param filtered bits that are one indicate rows that are not yet filtered. For these
     *                 the function has to check if it should be filtered, and in that case
     *                 for the given position in the result bitset a zero must be returned.
     * @return
     */
    public BitSet filterRows(DataTableColumn<?,U> data, List<String> ids, BitSet filtered);
    public double getApproximateFilterSpeed();

    /**
     * This method is expected to be blocking.
     *
     * In case that the FilterType requiresIntialization() returns true, then after changes of data in the
     * table (i.e. in the filtered column), the reinintFilter function is called, and afterwards the
     * filterRows function.
     *
     * @param dp
     * @param all_ids
     * @param changed_ids
     */
    public void reinitFilter(DataTableColumn<?,U> data, List<String> all_ids, List<String> changed_ids);

//    /**
//     * For certain filters it may be possible that the column first has
//     * to initialize specific datastructures asynchronously. In this case,
//     * the filter will return false in this function until the data is ready.
//     * Example would be the substructure filter and a structure column (loading
//     * fingerprints asynchronously).
//     *
//     * @return
//     */
//    public FilterState getFilterState();

//    public static enum FilterState {READY,UPDATING};
//
//    public void addFilterListener(FilterListener li);
//
//    public boolean removeFilterListener(FilterListener li);
//
//    public static interface FilterListener {
//        public void filterStateChanged(FilterState si);
//        public void filterConfigurationChanged();
//    }

}
