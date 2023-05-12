package tech.molecules.leet.datatable.dataprovider;

import java.util.Map;

public class DefaultNumericDataProvider extends HashMapBasedDataProvider<Double> {

    public DefaultNumericDataProvider(Map<String, Double> data) {
        super(data);
    }

}
