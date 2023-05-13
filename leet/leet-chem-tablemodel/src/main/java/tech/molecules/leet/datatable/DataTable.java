package tech.molecules.leet.datatable;

import org.apache.commons.lang3.tuple.Pair;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;

/**
 *
 * Selection Model: ..
 *
 *
 * Event Handling: Data columns have their own data provider that always
 * "immediately" provides them with data. However, data columns should not
 * themselves listen for changed in the data provider. Instead, data providers
 * can be registered in the DataTable, and then in case that a data provider
 * sends a data changed event, it notifies repaint events to all columns that
 * use the given data provider.
 *
 *
 *
 *
 */
public class DataTable {



    public enum TableStatus {READY,UPDATING};
    private TableStatus tableStatus = TableStatus.READY;

    public abstract class DataTableTask {
        public abstract void runTask();
        public void process() {
            tableStatus = TableStatus.UPDATING;
            runTask();
            tableStatus = TableStatus.READY;
        }
    }

    private BlockingQueue<DataTableTask> taskQueue;
    private Thread processingThread;

    public void initDataTableUpdateThread() {
        taskQueue = new LinkedBlockingQueue<>();
        processingThread = new Thread(() -> {
            while (true) {
                try {
                    DataTableTask task = taskQueue.take();
                    task.process();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        processingThread.start();
    }

    public void initSelectionModel() {
        this.selectionModel = new DataTableSelectionModel();
        this.selectionModel.addSelectionListener(new DataTableSelectionModel.SelectionListener() {
            @Override
            public void selectionStatusChanged(Collection<String> rows) {
                fireTableCellsChanged( rows.parallelStream().map( ri -> new int[]{ visibleKeysSortedPositions.get(ri),0}).collect(Collectors.toList()));
            }
        });
    }


    public DataTable() {
        this.initDataTableUpdateThread();
        this.initSelectionModel();
    }

    private List<DataTableColumn> columns = new ArrayList<>();

    private DataTableSelectionModel selectionModel;

    public DataTableSelectionModel getSelectionModel() {
        return this.selectionModel;
    }



    public List<String> getVisibleKeysSortedAt(int rows[]) {
        List<String> rows_a = new ArrayList<>();
        synchronized(this.visibleKeysSorted) {
            for (int zi = 0; zi < rows.length; zi++) {
                String rowid = this.getVisibleKeysSorted().get( rows[zi] );
                rows_a.add(rowid);
            }
        }
        return rows_a;
    }

    public List<DataTableColumn> getDataColumns() {
        synchronized (this.columns) {
            return new ArrayList<>(this.columns);
        }
    }

    public void addDataColumn(DataTableColumn dtc) {
        synchronized (this.columns) {
            this.columns.add(dtc);
            this.filters.put(dtc,new ArrayList<>());

            // add necessary listeners
            dtc.addColumnListener(new DataTableColumn.DataTableColumnListener() {
                @Override
                public void filteringChanged(DataTableColumn col) {
                    fireTableDataChanged();
                }

                @Override
                public void sortingChanged(DataTableColumn col) {
                    fireTableDataChanged();
                }

                @Override
                public void dataProviderChanged(DataTableColumn col, DataProvider newDP) {
                    fireTableDataChanged();
                }

                @Override
                public void visualizationChanged(DataTableColumn col) {
                    fireTableDataChanged();
                }
            });
        }
        fireTableStructureChanged();
    }

    public boolean removeDataColumn(DataTableColumn dtc) {
        synchronized (this.columns) {
            return this.columns.add(dtc);
        }
    }


    private List<String> allKeys = new ArrayList<>();
    private List<String> visibleKeysUnsorted = new ArrayList<>();
    private List<String> visibleKeysSorted = new ArrayList<>();

    //private Map<String,Integer> visibleKeyPositions = new HashMap<>();

    public List<String> getVisibleKeysUnsorted() {
        synchronized (visibleKeysUnsorted) {
            return new ArrayList<>(visibleKeysUnsorted);
        }
    }

    public List<String> getVisibleKeysSorted() {
        synchronized (visibleKeysSorted) {
            return new ArrayList<>(visibleKeysSorted);
        }
    }

    public List<String> getAllKeys() {
        synchronized (allKeys) {
            return new ArrayList<>(allKeys);
        }
    }

    public void setAllKeys(List<String> keys) {
        synchronized (visibleKeysUnsorted) {
            this.allKeys = new ArrayList<>(keys);
            this.taskQueue.add(new FullUpdateTask(null,null));
        }
    }

    private Map<String, Integer> visibleKeysSortedPositions = new HashMap<>();

    private void reinitVisibleKeysSortedPositions() {
        this.visibleKeysSortedPositions.clear();
        for(int zi = 0; zi<this.visibleKeysUnsorted.size(); zi++) {
            this.visibleKeysSortedPositions.put(this.visibleKeysSorted.get(zi),zi);
        }
    }

    private void setVisibleKeysSorted(List<String> vk) {
        synchronized(visibleKeysSorted) {
            this.visibleKeysSorted = vk;
            this.reinitVisibleKeysSortedPositions();
        }
    }

    private Map<DataTableColumn,List<DataFilter>> filters = new HashMap<>();

    public boolean removeFilter(DataTableColumn dtc, AbstractCachedDataFilter fi) {
        boolean removed = false;
        synchronized(this.columns) {
            removed = this.filters.get(dtc).remove(fi);
        }
        if(removed) {
            // this might work to recombine all remaining filter bitsets:
            this.taskQueue.add(new FullUpdateTask(new HashSet<>(),new ArrayList<>()));
        }
        return removed;
    }

    public void addFilter(DataTableColumn dtc, DataFilter fi) {
        synchronized(this.columns) {
            this.filters.get(dtc).add(fi);
        }
        fi.addFilterListener(new DataFilter.FilterListener() {
            @Override
            public void filterChanged() {
                taskQueue.add(new FullUpdateTask(Collections.singletonMap(fi,null).keySet(),null));
            }
        });
        this.taskQueue.add(new FullUpdateTask(Collections.singletonMap(fi,null).keySet(),null));
    }

    private List<Pair<DataTableColumn,DataSort>> sorts = new ArrayList<>();


    /**
     * The separate BitSets are wrt allKeys
     */
    private Map<DataFilter,BitSet> filterData = new ConcurrentHashMap<>();

    /**
     * This defines the positions
     */
    //private Map<String,Integer> visibleKeysPositions = new HashMap<>();

    /**
     * NOTES: keysDataChanged does only affect the update of the filters
     *
     * 1. loop over keysAll to find positions in there
     * 2. reinit filters that require reinit
     * 3. update filtering data for all changed keys
     * 4. combine all filtering data and create visibleKeys
     * 5. sort visibleKeys according to the sorts and create sortedVisibleKeys.
     *
     *
     */
    private class FullUpdateTask extends DataTableTask {

        private Set<? extends DataFilter> updatedFilters;
        private List<String> keysDataChanged;

        /**
         * NOTE!! updatedFilter == null means all filters
         * NOTE!! updatedKeys == null means all data
         *
         * @param updatedFilters
         * @param keysDataChanged
         */
        public  FullUpdateTask(Set<? extends DataFilter> updatedFilters, List<String> keysDataChanged) {
            this.updatedFilters  = updatedFilters;
            this.keysDataChanged = keysDataChanged;
        }

        @Override
        public void runTask() {
            synchronized(allKeys) {

                if(keysDataChanged==null) {
                    // in this case we have to consider all keys..
                    keysDataChanged = new ArrayList<>(allKeys);
                }

                Map<String,Integer> allRowPos = new HashMap<>();
                for(int zi=0;zi<allKeys.size();zi++) {
                    allRowPos.put(allKeys.get(zi),zi);
                }

                // refilter the keys with data changed..
                List<Thread> threads_i = new ArrayList<>();
                for(Map.Entry<DataTableColumn,List<DataFilter>> efi : filters.entrySet()) {

                    DataTableColumn ci = efi.getKey();
                    for(DataFilter fi : efi.getValue()) {

                        // check if this filter is updated
                        if(updatedFilters!=null && !updatedFilters.contains(fi)) {
                            // in this case we can skip..
                            continue;
                        }

                        Runnable ri = new Runnable() {
                            @Override
                            public void run() {
                                // check if reinit is required..
                                if(fi.getDataFilterType().requiresInitialization()) {
                                    // to ensure that we wait for potentially cngoing reinits we get the monitor
                                    // for the object. We always to this for reinit
                                    synchronized(fi) {
                                        fi.reinitFilter(ci, allKeys, keysDataChanged);
                                    }
                                }

                                // compute filtering
                                BitSet bsi = fi.filterRows(ci, keysDataChanged, new BitSet(keysDataChanged.size()));


                                // now recombine with existing data, i.e. update existing data
                                BitSet f_old = new BitSet();
                                if(filterData.containsKey(fi)) {
                                    f_old = filterData.get(fi);
                                }

                                for(int zi=0;zi<keysDataChanged.size();zi++) {
                                    f_old.set( allRowPos.get( keysDataChanged.get(zi) ) , bsi.get(zi) );
                                }
                                filterData.put(fi,f_old);
                            }
                        };
                        Thread ti = new Thread(ri);
                        ti.start();
                        threads_i.add(ti);
                    }
                }
                // wait for all threads to finish:
                System.out.println("[DEBUG] separate refilter threads started -> wait");
                threads_i.stream().forEach( ti -> {
                    try {
                        ti.join();
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                });
                System.out.println("[DEBUG] separate refilter threads started -> wait -> ALL THREADS FINISHED!");

                // Now combine all filtering data
                BitSet bs_filtered = new BitSet();
                for(BitSet bi : filterData.values()) {
                    bs_filtered.or(bi);
                }

                System.out.println("[DEBUG] total filtered: "+bs_filtered.cardinality()+ " / "+allKeys.size());

                // now update the visibleKeys
                List<String> newVisibleKeys = new ArrayList<>( (allKeys.size()-bs_filtered.cardinality()) );
                for(int zi=0;zi<allKeys.size();zi++) {
                    if(!bs_filtered.get(zi)) {
                        newVisibleKeys.add( allKeys.get(zi) );
                    }
                }

                // now recompute the order
                synchronized(visibleKeysUnsorted) {
                    visibleKeysUnsorted = newVisibleKeys;

                    if(sorts.size()>0) {
                        List<String> newVisibleKeysSorted = new ArrayList<>(visibleKeysUnsorted);
                        // Sort the objects based on the list of comparators

                        newVisibleKeysSorted.sort((a, b) -> {
                            for (Pair<DataTableColumn, DataSort> comparator : sorts) {
                                DataTableColumn.CellValue xa = comparator.getLeft().getValue(a);
                                DataTableColumn.CellValue xb = comparator.getLeft().getValue(b);
                                int result = comparator.getRight().compare(xa.val, xb.val);
                                if (result != 0) {
                                    return result;
                                }
                            }
                            return 0;
                        });
                        synchronized (visibleKeysSorted) {
                            //visibleKeysSorted = newVisibleKeysSorted;
                            setVisibleKeysSorted(newVisibleKeysSorted);
                        }
                    }
                    else {
                        //visibleKeysSorted = new ArrayList<>(visibleKeys);
                        setVisibleKeysSorted(new ArrayList<>(visibleKeysUnsorted));
                    }

                }
            }

            fireTableDataChanged();
        }
    }

    public void setDataSort(List<Pair<DataTableColumn,DataSort>> sort) {
        synchronized (this.sorts) {
            this.sorts = sort;
        }
        this.taskQueue.add(new FullUpdateTask(new HashSet<>(),new ArrayList<>()));
    }


//    private class UpdateRowFilteringTask implements DataTableTask {
//        @Override
//        public void process() {
//            synchronized(visibleKeys) {
//                // TODO: implement filtering
//                //visibleKeys = new ArrayList<>(allKeys);
//                setVisibleKeys(allKeys);
//            }
//        }
//    }

    private List<DataTableListener> listeners = new ArrayList<>();

    public void addDataTableListener(DataTableListener li) {
        listeners.add(li);
    }

    public boolean removeDataTableListener(DataTableListener li) {
        return listeners.remove(li);
    }


    public static class CellState {
        public final Color backgroundColor;
        public final List<Color> selectionColors;
        public CellState(Color backgroundColor, List<Color> selectionColors) {
            this.backgroundColor = backgroundColor;
            this.selectionColors = selectionColors;
        }
    }

    public DataTableColumn.CellValue getValue(int row, int col) {
        return getDataColumns().get(col).getValue(this.visibleKeysSorted.get(row));
    }

    public DataTable.CellState getCellState(int row, int col) {
        DataTableColumn.CellValue cv = getDataColumns().get(col).getValue(this.visibleKeysSorted.get(row));
        //return new CellState(cv.colBG, Arrays.asList(new Color[]{Color.red,Color.blue}));

        List<DataTableSelectionModel.SelectionType> sti = getSelectionModel().getSelectionTypesForRow(this.visibleKeysSorted.get(row));
        List<Color> selectionColors = sti.stream().map(xi -> xi.getColor()).collect(Collectors.toList());
        return new CellState(cv.colBG,selectionColors);
    }


    private void fireTableStructureChanged() {
        for(DataTableListener li : listeners) {
            li.tableStructureChanged();
        }
    }

    private void fireTableCellsChanged(List<int[]> cells) {
        for(DataTableListener li : listeners) {
            li.tableCellsChanged(cells);
        }
    }

    private void fireTableDataChanged() {
        for(DataTableListener li : listeners) {
            li.tableDataChanged();
        }
    }

    public static interface DataTableListener {
        /**
         * Means columns were added / removed
         */
        public void tableDataChanged();
        public void tableStructureChanged();
        /**
         *
         * @param cells entries are {row,col}
         */
        public void tableCellsChanged(List<int[]> cells);
    }

}
