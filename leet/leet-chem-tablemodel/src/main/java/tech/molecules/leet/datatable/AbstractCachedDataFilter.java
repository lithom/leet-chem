package tech.molecules.leet.datatable;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public abstract class AbstractCachedDataFilter<T> implements DataFilter<T> {

    /**
     *
     * @param data
     * @param ids
     * @param filtered bits that are one indicate rows that are not yet filtered. For these
     *                 the function has to check if it should be filtered, and in that case
     *                 for the given position in the result bitset a one must be returned.
     * @return
     */
    public BitSet filterRows(DataTableColumn<?,T> data, List<String> ids, BitSet filtered) {
        BitSet bsi = (BitSet) filtered.clone();
        for(int zi=0;zi<ids.size();zi++) {
            if(!filtered.get(zi)) {
                //if(filterRow(data.getData(ids.get(zi)))) {
                if(cached.get(ids.get(zi))) {
                    bsi.set(zi,true);
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

    /**
     * To ensure that we run only one reinitFilter at the time,
     * by convention this function must be called synchronized,
     * i.e. we also ensure this while calling the function.
     *
     * @param data
     * @param all_ids
     */
    @Override
    public synchronized void reinitFilter(DataTableColumn<?,T> data, List<String> all_ids, List<String> changed) {
        this.state = FilterState.UPDATING;
        cached = new ConcurrentHashMap<>();
        Runnable ri = new Runnable() {
            @Override
            public void run() {
                changed.parallelStream().forEach( si -> {
                    boolean bi = filterRow(data.getValue(si).val);
                    cached.put(si,bi);
                } );

            }
        };
        Thread ti = new Thread(ri);
        ti.start();
        // NOTE: by convention this function is blocking, so we wait..
        try {
            ti.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        state = FilterState.READY;
    }

}
