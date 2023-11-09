package tech.molecules.leet.datatable.dataprovider;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataProvider;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class SerializingHashMapBasedDataProvider <T> implements DataProvider<T> {

    private Map<String,String> data;
    private Function<T,String> serializer;
    private Function<String,T> deserializer;

    private Map<String,String> serializeMap(Map<String,T> data_a) {
        Map<String,String> serialized = new HashMap<>();
        data_a.entrySet().forEach( xi -> serialized.put( xi.getKey() , serializer.apply(xi.getValue()) ) );
        return serialized;
    }

    //public SerializingHashMapBasedDataProvider(Map<String,T> data, Map<String,String> data_raw, Function<T,String> serializer, Function<String,T> deserializer) {
    public SerializingHashMapBasedDataProvider(Map<String,T> data, Function<T,String> serializer, Function<String,T> deserializer) {
        this.serializer = serializer;
        this.deserializer = deserializer;
        this.data = new HashMap<>( serializeMap(data) );
    }

    public void addData(Map<String,T> more_data) {
        this.data.putAll( serializeMap(more_data) );
        listenerHelper.fireDataChanged(new ArrayList<>(more_data.keySet()));
    }

    public void addRawData(Map<String,String> more_data) {
        this.data.putAll(more_data);
        listenerHelper.fireDataChanged(new ArrayList<>(more_data.keySet()));
    }

    public void addRawData(List<Pair<String,String>> more_data) {
        more_data.forEach( xi -> this.data.put(xi.getLeft(),xi.getRight()) );
        listenerHelper.fireDataChanged(new ArrayList<>(more_data.stream().map(xi -> xi.getLeft()).collect(Collectors.toList())));
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
        return this.deserializer.apply(this.data.get(key));
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