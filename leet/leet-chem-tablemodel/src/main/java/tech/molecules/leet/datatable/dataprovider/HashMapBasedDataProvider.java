package tech.molecules.leet.datatable.dataprovider;

import tech.molecules.leet.datatable.DataProvider;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HashMapBasedDataProvider<T> implements DataProvider<T> {

    private Map<String,T> data;

    public HashMapBasedDataProvider(Map<String,T> data) {
        this.data = new HashMap<>(data);
    }

    public void addData(Map<String,T> more_data) {
        this.data.putAll(more_data);
        listenerHelper.fireDataChanged(new ArrayList<>(more_data.keySet()));
    }

    public void removeData(List<String> keysToRemove) {
        for(String ki : keysToRemove) {
            this.data.remove(ki);
        }
        listenerHelper.fireDataChanged(keysToRemove);
    }

    @Override
    public List<String> getAllEntries() {
        return new ArrayList<>(this.data.keySet());
    }

    @Override
    public T getData(String key) {
        return this.data.get(key);
    }

    private DataProviderListenerHelper listenerHelper = new DataProviderListenerHelper();

    @Override
    public void addDataProviderListener(DataProviderListener li) {
        listenerHelper.addDataProviderListener(li);
    }

    @Override
    public boolean removeDataProviderListener(DataProviderListener li) {
        return listenerHelper.removeDataProviderListener(li);
    }
}
