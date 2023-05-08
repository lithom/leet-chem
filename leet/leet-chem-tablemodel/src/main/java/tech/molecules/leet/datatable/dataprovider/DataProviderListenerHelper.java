package tech.molecules.leet.datatable.dataprovider;

import tech.molecules.leet.datatable.DataProvider;

import java.util.List;

public class DataProviderListenerHelper {

    private List<DataProvider.DataProviderListener> listeners;

    public void addDataProviderListener(DataProvider.DataProviderListener li) {
        this.listeners.add(li);
    }

    public boolean removeDataProviderListener(DataProvider.DataProviderListener li) {
        return this.listeners.remove(li);
    }

    public void fireDataChanged(List<String> keys) {
        for(DataProvider.DataProviderListener li : listeners) {
            li.dataChanged(keys);
        }
    }

}
