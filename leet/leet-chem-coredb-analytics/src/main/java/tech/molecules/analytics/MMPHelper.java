package tech.molecules.analytics;

import tech.molecules.chem.coredb.AssayResult;

import java.util.*;

public class MMPHelper {

    public static <T extends MMPInstance> Map<MMPTransformation,List<T>> sortMMPsByTransformation(List<T> mmps) {
        Map<MMPTransformation,List<T>> sorted = new HashMap<>();
        for(T mmpi : mmps) {
            MMPTransformation tfi = mmpi.getTransformation();
            if(!sorted.containsKey(tfi)) {sorted.put(tfi,new ArrayList<>());}
            sorted.get(tfi).add(mmpi);
        }
        return sorted;
    }


    /**
     *
     * @param mmps
     * @return
     * @param <T>
     */
    public static <T extends MMPInstance> Map<String,Map<MMPTransformation,List<T>>> sortMMPsByTransformationWithoutDirection(List<T> mmps) {
        Map<String,Map<MMPTransformation,List<T>>> sorted = new HashMap<>();
        for(T mmpi : mmps) {
            MMPTransformation tfi = mmpi.getTransformation();
            String twd = tfi.getTransformationIdWithoutDirection();
            if(!sorted.containsKey(twd)) {sorted.put(twd,new HashMap<>());}
            if(!sorted.get(twd).containsKey(tfi)){sorted.get(twd).put(tfi,new ArrayList<>());}
            sorted.get(twd).get(tfi).add(mmpi);
        }
        return sorted;
    }

    public static NumericalMMPInstance createNumericalMMPInstance(MMPInstance mmp , String attribute, Map<Long, AssayResult> assayResults) {

        double values_a[] = mmp.getFragmentDecompositionA().getAssayResultIDs().stream().filter(xi -> assayResults.get(xi).getData(attribute)!=null).mapToDouble( xi -> assayResults.get(xi).getData(attribute).getAsDouble() ).toArray();
        double values_b[] = mmp.getFragmentDecompositionB().getAssayResultIDs().stream().filter(xi -> assayResults.get(xi).getData(attribute)!=null).mapToDouble( xi -> assayResults.get(xi).getData(attribute).getAsDouble() ).toArray();
        double mean_a = Arrays.stream(values_a).summaryStatistics().getAverage();
        double mean_b = Arrays.stream(values_b).summaryStatistics().getAverage();

        NumericalMMPInstance mmp2 = new NumericalMMPInstanceImpl(mmp.getFragmentDecompositionA(),mmp.getFragmentDecompositionB(),attribute,mean_a,mean_b);
        return mmp2;
    }

}
