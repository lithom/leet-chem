package tech.molecules.leet.datatable;

import javax.swing.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public abstract class AbstractDataFilter<T> implements DataFilter<T> {

    /**
     *
     * @param data
     * @param ids
     * @param filtered bits that are one indicate rows that are not yet filtered. For these
     *                 the function has to check if it should be filtered, and in that case
     *                 for the given position in the result bitset a zero must be returned.
     * @return
     */
    public BitSet filterRows(DataProvider<T> data, List<String> ids, BitSet filtered) {
        BitSet bsi = (BitSet) filtered.clone();
        for(int zi=0;zi<ids.size();zi++) {
            if(filtered.get(zi)) {
                //if(filterRow(data.getData(ids.get(zi)))) {
                if(cached.get(ids.get(zi))) {
                    bsi.set(zi,false);
                }
            }
        }
        return bsi;
    }

    /**
     * Return true if the filter should remove the given object
     *
     * @param vi
     * @return
     */
    public abstract boolean filterRow(T vi);

    private FilterState state = FilterState.UPDATING;

    /**
     * True value means should be filtered by this filter
     */
    private Map<String,Boolean> cached ;

    @Override
    public synchronized void setupFilter(DataProvider<T> dp, List<String> ids) {
        state = FilterState.UPDATING;
        fireFilterStateChanged();
        cached = new ConcurrentHashMap<>();
        Runnable ri = new Runnable() {
            @Override
            public void run() {
                ids.parallelStream().forEach( si -> {
                    boolean bi = filterRow(dp.getData(si));
                    cached.put(si,bi);
                } );
                state = FilterState.READY;
                fireFilterStateChanged();
            }
        };
        Thread ti = new Thread(ri);
        ti.start();
    }

    private List<FilterListener> listeners = new ArrayList<>();

    private void fireFilterStateChanged() {
        for(FilterListener li : listeners) {
            li.filterStateChanged(this.state);
        }
    }
    public void addFilterListener(FilterListener li) {
        this.listeners.add(li);
    }
    public boolean removeFilterListener(FilterListener li) {
        return this.listeners.remove(li);
    }

}
