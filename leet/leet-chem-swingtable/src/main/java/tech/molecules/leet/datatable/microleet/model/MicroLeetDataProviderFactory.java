package tech.molecules.leet.datatable.microleet.model;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataProvider;

import java.util.List;

public interface MicroLeetDataProviderFactory<C> {

    public DataProvider<C> initDataProvider(List<Pair<String,String>> data);

}
