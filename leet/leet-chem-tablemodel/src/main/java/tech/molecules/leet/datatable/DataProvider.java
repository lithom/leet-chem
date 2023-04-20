package tech.molecules.leet.datatable;

import java.util.List;

public interface DataProvider<T> {

    public List<String> getAllEntries();
    public T getData(String key);

    public static interface DataProviderListener {
        public void dataChanged(List<String> keysChanged);
    }

    public void addDataProviderListener(DataProviderListener li);
    public boolean removeDataProviderListener(DataProviderListener li);

}
