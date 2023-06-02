package tech.molecules.analytics;

import java.util.List;
import java.util.Map;

public interface MMPDatabase {

    public List<NumericalMMPTransformationStatistics> fetchMMPsForAssay(int assay_id, String attribute, int min_N);

    public List<NumericalMMPTransformationStatistics> searchMMPs(MMPNumericQuery query);

    public Map<MMPTransformation,List<MMPInstance>> fetchMMPInstances(MMPQuery query);

    public Map<NumericalMMPTransformationStatistics,List<NumericalMMPInstance>> fetchMMPInstances(MMPNumericQuery query);


}
