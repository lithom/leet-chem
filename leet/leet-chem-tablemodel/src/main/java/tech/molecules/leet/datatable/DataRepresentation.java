package tech.molecules.leet.datatable;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @param <A> original representation
 * @param <B> alternative representation
 */
public interface DataRepresentation<A,B> {

    public String getName();

    public B evaluate(A original);
    public Class<B> getRepresentationClass();

    default public DataProvider<B> getAsDataProvider(DataProvider<A> dp) {
        return new DataProvider<B>() {
            {dp.addDataProviderListener(new DataProviderListener() {
                @Override
                public void dataChanged(List<String> keysChanged) {
                    fireDataChanged(keysChanged);
                }
            });}
            @Override
            public List<String> getAllEntries() {
                return dp.getAllEntries();
            }

            @Override
            public B getData(String key) {
                return evaluate( dp.getData(key) );
            }

            private List<DataProviderListener> listeners = new ArrayList<>();

            @Override
            public void addDataProviderListener(DataProviderListener li) {
                this.listeners.add(li);
            }

            @Override
            public boolean removeDataProviderListener(DataProviderListener li) {
                return this.listeners.remove(li);
            }

            private void fireDataChanged(List<String> keysChanged) {
                for(DataProviderListener li : this.listeners) {
                    li.dataChanged(keysChanged);
                }
            }
        };
    }

}
