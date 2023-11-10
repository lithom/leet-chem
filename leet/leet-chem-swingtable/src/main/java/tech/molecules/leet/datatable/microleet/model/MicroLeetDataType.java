package tech.molecules.leet.datatable.microleet.model;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.DataProvider;

import java.util.List;
import java.util.stream.Collectors;

public class MicroLeetDataType<C> {


    public final String name;

    public final MicroLeetDataValueSerializer<C> serializer;

    public final MicroLeetDataProviderFactory dataProviderFactory;

    public final MicroLeetDataColumnFactory dataColumnFactory;

    public MicroLeetDataType(String name,
                             MicroLeetDataValueSerializer<C> serializer,
                             MicroLeetDataProviderFactory dataProviderFactory,
                             MicroLeetDataColumnFactory dataColumnFactory) {
        this.name = name;
        this.serializer = serializer;
        this.dataProviderFactory = dataProviderFactory;
        this.dataColumnFactory = dataColumnFactory;
    }

    /**
     * Checks for all supplied strings if they can be interpreted.
     *
     * @param data
     * @return
     */
    public double checkCompatibility(List<String> data) {
        double count = 0;
        for(String si : data) {
            boolean success = this.serializer.initFromString(si) != null;
            if(success) {count += 1.0;}
        }
        return count / data.size();
    }

    public DataProvider<C> loadDataRaw(List<Pair<String,String>> data) {
        List<Pair<String,C>> parsed = data.parallelStream().map( xi -> Pair.of(xi.getLeft(), serializer.initFromString(xi.getRight()))).collect(Collectors.toList());
        DataProvider<C> dp =  dataProviderFactory.initDataProvider( parsed );
        return dp;
    }

}
