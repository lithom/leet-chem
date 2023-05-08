package tech.molecules.analytics;

import java.util.List;

public interface MMPDatabase {

    public void storeFragmentDecompositions(List<MMPFragmentDecomposition> decompositions);

    public void storeNumericalMMPTransformationStatistics(List<NumericalMMPTransformationStatistics> decompositions);

    public List<NumericalMMPTransformationStatistics> fetchMMPsForAssay(int assay_id, String attribute, int min_N);


}
