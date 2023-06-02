package tech.molecules.analytics;

import java.util.List;

public interface MMPDatabaseWriter {

    public void storeFragmentDecompositions(List<MMPFragmentDecomposition> decompositions);

    public void storeNumericalMMPTransformationStatistics(List<NumericalMMPTransformationStatistics> decompositions);


}
