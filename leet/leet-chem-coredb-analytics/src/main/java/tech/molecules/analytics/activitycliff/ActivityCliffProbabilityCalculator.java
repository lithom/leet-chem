package tech.molecules.analytics.activitycliff;

import tech.molecules.analytics.*;
import tech.molecules.chem.coredb.Assay;
import tech.molecules.chem.coredb.AssayResult;
import tech.molecules.leet.chem.util.Parallelizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ActivityCliffProbabilityCalculator {

    private ActivityCliffDefinition activityCliffDefinition;

    public ActivityCliffProbabilityCalculator(ActivityCliffDefinition activityCliffDefinition) {
        this.activityCliffDefinition = activityCliffDefinition;
    }

    public static ActivityCliffTransformationData.ACData computeActivityCliffProbability(ActivityCliffDefinition acd, List<NumericalMMPInstance> numericalMMPInstances) {
        int totalInstances = numericalMMPInstances.size();
        int activityCliffInstances = 0;
        int aHighBLow = 0;
        int bHighALow = 0;

        for (NumericalMMPInstance instance : numericalMMPInstances) {
            double v1 = instance.getMeanA();
            double v2 = instance.getMeanB();

            int ac = acd.computeActivityCliff(v1,v2);
            if (ac!=0) {
                activityCliffInstances++;
                if(ac>0) {aHighBLow++;}
                else{bHighALow++;}
            }
        }

        return new ActivityCliffTransformationData.ACData( (double) activityCliffInstances / totalInstances , numericalMMPInstances.size(),
                aHighBLow,bHighALow);
    }


    /**
     * NOTE: Direction of MMP Transformation is canonized in the result of this function, i.e. we will not have
     *       both transformations A -> B and B -> A in the results, only one of both.
     *
     * @param consideredAssays
     * @param assay_results_provider
     * @param numericalAttribute
     * @param cliffdefs
     * @param numCores
     */
    public static void processAssays(List<Assay> consideredAssays, Function<Assay,List<AssayResult>>assay_results_provider, String numericalAttribute, List<ActivityCliffDefinition> cliffdefs, int numCores) {
        Map<MMPTransformation,List<NumericalMMPInstance>> mmps_sortedByTransformation = new HashMap<>();

        List<Runnable> tasks = new ArrayList<>();
        for(int zi=0;zi<consideredAssays.size();zi++) {
            int fzi = zi;
            Runnable ri = new Runnable() {
                @Override
                public void run() {
                    List<NumericalMMPInstance> mmps_i = processAssay(consideredAssays.get(fzi),assay_results_provider,numericalAttribute,cliffdefs);
                    Map<String,Map<MMPTransformation,List<NumericalMMPInstance>>> mmps_sorted = MMPHelper.sortMMPsByTransformationWithoutDirection(mmps_i);

                    // sort by mmp transformation
                    Map<MMPTransformation,List<NumericalMMPInstance>> mmps_sortedByTransformation_i = new HashMap<>();

                    for(String tf_a : mmps_sorted.keySet()) {
                        List<NumericalMMPInstance> allSameDirectionMMPs = new ArrayList<>();
                        // now we change direction of mmps that are in "opposite" direction:
                        MMPTransformation tf_canonical    = null;
                        MMPTransformation tf_toBeReversed = null;
                        MMPTransformation tfi = mmps_sorted.get(tf_a).keySet().iterator().next();
                        MMPTransformation tfi2 = tfi.getInverseTransformation();
                        if(tfi.compareTo(tfi2)>0) {
                            tf_canonical = tfi; tf_toBeReversed = tfi2;
                        }
                        else {
                            tf_canonical = tfi2; tf_toBeReversed = tfi;
                        }
                        // Now do..
                        if(mmps_sorted.get(tf_a).containsKey(tf_canonical)) {
                            allSameDirectionMMPs.addAll(mmps_sorted.get(tf_a).get(tf_canonical));
                        }
                        if(mmps_sorted.get(tf_a).containsKey(tf_toBeReversed)) {
                            allSameDirectionMMPs.addAll(mmps_sorted.get(tf_a).get(tf_canonical).stream().map( xi -> (NumericalMMPInstance) xi.getInverseMMPInstance() ).collect(Collectors.toList()));
                        }
                        // put in..
                        mmps_sortedByTransformation_i.put( tf_canonical , allSameDirectionMMPs);
                    }

                    synchronized(mmps_sortedByTransformation) {
                        for(MMPTransformation tfi : mmps_sortedByTransformation_i.keySet()) {
                            if(!mmps_sortedByTransformation.containsKey(tfi)) {mmps_sortedByTransformation.put(tfi,new ArrayList<>());}
                            mmps_sortedByTransformation.get(tfi).addAll(mmps_sortedByTransformation_i.get(tfi));
                        }
                    }
                }
            };
        }
        try {
            Parallelizer.computeParallelBlocking(tasks,numCores,1);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        System.out.println("[INFO] All MMP Computations done!");

        // now really compute ACData for transformations
        List<ActivityCliffTransformationData> datapoints = new ArrayList<>();

        for(ActivityCliffDefinition acdi : cliffdefs) {
            for (MMPTransformation tfi : mmps_sortedByTransformation.keySet()) {
                List<NumericalMMPInstance> mmps_i = mmps_sortedByTransformation.get(tfi);
                ActivityCliffTransformationData.ACData acdata = computeActivityCliffProbability(acdi,mmps_i);
                datapoints.add(new ActivityCliffTransformationDataImpl(tfi,mmps_i,acdi,acdata));
            }
        }


    }

    public static List<NumericalMMPInstance> processAssay(Assay assay, Function<Assay,List<AssayResult>>assay_results_provider, String numericalAttribute, List<ActivityCliffDefinition> cliffdefs) {
        List<AssayResult> results = assay_results_provider.apply(assay);

        // 1. compute mmp fragment decomps
        List<MMPFragmentDecomposition> decomps = MMPComputationTool.computeMMPFragmentDecompositions(results,14,0.3,3,1);

        // 2. compute mmps and sort by transformation
        List<MMPInstance> mmps = MMPComputationTool.computeMMPs(decomps);
        Map<MMPTransformation,List<MMPInstance>> sortedMMPs = MMPHelper.sortMMPsByTransformation(mmps);
        System.out.println("Transformations: "+sortedMMPs.size());

        // 3. create numerical mmps:
        List<NumericalMMPInstance> numerical_mmps = new ArrayList<>();
        Map<Long,AssayResult> all_results_by_id = new HashMap<>();
        for(AssayResult ri : results) {all_results_by_id.put(ri.getId(),ri);}
        for( MMPInstance mpi : mmps) {
            NumericalMMPInstance nni = MMPHelper.createNumericalMMPInstance(mpi,numericalAttribute,all_results_by_id);
            numerical_mmps.add(nni);
        }

        // 4. return results..
        return numerical_mmps;
    }


}