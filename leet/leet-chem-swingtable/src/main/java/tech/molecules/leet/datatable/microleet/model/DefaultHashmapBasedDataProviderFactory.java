package tech.molecules.leet.datatable.microleet.model;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.dataprovider.HashMapBasedDataProvider;
import tech.molecules.leet.datatable.dataprovider.SerializingHashMapBasedDataProvider;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class DefaultHashmapBasedDataProviderFactory<C> implements MicroLeetDataProviderFactory<C> {

    private MicroLeetDataValueSerializer<C> serializer;

    public DefaultHashmapBasedDataProviderFactory(MicroLeetDataValueSerializer<C> serializer) {
        this.serializer = serializer;
    }

    @Override
    public DataProvider<C> initDataProvider(List<Pair<String, String>> data) {
        //Map<String,C> hm = new HashMap<>();
        //data.stream().forEach( xi -> hm.put(xi.getLeft(),xi.getRight()) );
        Map<String,String> data_initialized = new HashMap<>();
        for(Pair<String,String> di : data) {
            if(di.getRight()!=null && di.getRight().length()>0) {
                data_initialized.put(di.getLeft(), fullInit(di.getRight()));
            }
        }
        SerializingHashMapBasedDataProvider<C> sdp = new SerializingHashMapBasedDataProvider<C>(new HashMap<>(), xi -> this.serializer.serializeToString(xi) , xi -> this.serializer.initFromString(xi) );
        sdp.addRawData(data_initialized);
        return sdp;
    }

    /**
     * deserializes and then serializes data.
     */
    private String fullInit(String data) {
        return this.serializer.serializeToString( this.serializer.initFromString(data) );
    }

}
