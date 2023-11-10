package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class DefaultNumericDecompositionProvider extends DefaultDecompositionProvider implements NumericDecompositionProvider {

    private Map<String,Double> data;


    /**
     * Decompositions must be the fully decomposed molecules.
     *
     * @param data
     * @param decompositions
     */
    public DefaultNumericDecompositionProvider(Map<String, Double> data, List<String> labels, List<Pair<String,Part>> decompositions) {
        super(labels, decompositions);
        this.data = data;
    }

    public DefaultNumericDecompositionProvider(Map<String, Double> data, DecompositionProvider provider) {
        super(provider.getAllLabels(), provider.getAllDecompositions());
        this.data = data;
    }



    @Override
    public DescriptiveStats.Stats getStatsForStructures(List<Pair<String,Part>> p) {
        //Map<Pair<String,Part>,Double> data_i = new HashMap<>();
        List<Double> v = new ArrayList<>();
        for(Pair<String,Part> pi : p) {
            v.add( this.data.get(pi.getLeft()) );
        }
        return DescriptiveStats.computeStats(v);
    }

}
