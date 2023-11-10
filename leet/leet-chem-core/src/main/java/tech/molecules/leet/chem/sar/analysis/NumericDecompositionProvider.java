package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

public interface NumericDecompositionProvider extends DecompositionProvider {
    public DescriptiveStats.Stats getStatsForStructures(List<Pair<String,Part>> p);

}
