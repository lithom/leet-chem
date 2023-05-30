package tech.molecules.analytics;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.AssayResult;
import tech.molecules.chem.coredb.AssayResultQuery;
import tech.molecules.chem.coredb.sql.DBAssayResult;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;
import tech.molecules.leet.chem.shredder.FragmentDecompositionShredder;
import tech.molecules.leet.chem.util.Parallelizer;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class MMPComputationTool {

    public static List<MMPFragmentDecomposition> computeMMPFragmentDecompositions(Connection conn, int assay_id,
                                                                                      int max_fragment_size, double max_relative_fragment_size,
                                                                                      int min_extension_size,
                                                                                      int max_splits) throws SQLException {

        AssayResultQuery arq = new AssayResultQuery(assay_id,null,null,null);
        List<AssayResult> results = DBAssayResult.searchAssayResults(conn,arq);

        return computeMMPFragmentDecompositions(results,
                max_fragment_size,max_relative_fragment_size,
                min_extension_size,
                max_splits);
    }

    public static List<MMPInstance> computeMMPs( List<MMPFragmentDecomposition> decompositions ) {
        Map<String,List<MMPFragmentDecomposition>> by_remainder = new HashMap<>();
        for(MMPFragmentDecomposition di : decompositions) {
            if(!by_remainder.containsKey(di.getRemainderIDCode())) {by_remainder.put(di.getRemainderIDCode(),new ArrayList<>());}
            by_remainder.get(di.getRemainderIDCode()).add(di);
        }

        List<MMPInstance> mmps = new ArrayList<>();
        for(String remi : by_remainder.keySet()) {
            List<MMPFragmentDecomposition> dci = by_remainder.get(remi);
            if(dci.size()>=2) {
                for(int zi=0;zi<dci.size()-1;zi++) {
                    for(int zj=zi+1;zj<dci.size();zj++) {
                        MMPInstance mmpi = new MMPInstanceImpl(dci.get(zi),dci.get(zj));
                        mmps.add(mmpi);
                    }
                }
            }
        }
        return mmps;
    }

    /**
     * NOTE! It is possible that due to symmetry of molecules we get multiple MMPFragmentDecompositions with
     * the same id from the decomposition of a single structure. This function will only return one single
     * version of multiple equal decompositions per structure.
     *
     *
     * @param results
     * @param max_fragment_size
     * @param max_relative_fragment_size
     * @param min_extension_size
     * @param max_splits
     * @return
     */
    public static List<MMPFragmentDecomposition> computeMMPFragmentDecompositions(List<AssayResult> results,
                                                                                      int max_fragment_size, double max_relative_fragment_size,
                                                                                      int min_extension_size,
                                                                                      int max_splits) {

        if(results.stream().mapToInt(ri -> ri.getAssay().getId()).distinct().count() > 1 ) {
            throw new RuntimeException("computeMMPFragmentDecompositions(..) : all AssayResult objects must be from the same assay");
        }

        int assay_id = results.get(0).getAssay().getId();

        Map<String,List<AssayResult>> sortedResults = new HashMap<>();
        Map<String,String> idcodeToMolid = new HashMap<>();
        results.stream().forEach(ri -> {
            String molid_i = ri.getTube().getBatch().getCompound().getId();
            if(!sortedResults.containsKey(molid_i)) {sortedResults.put(molid_i,new ArrayList<>());}
            sortedResults.get(molid_i).add(ri);
            idcodeToMolid.put(ri.getTube().getBatch().getCompound().getMolecule()[0],molid_i);
        });

        Map<String,List<FragmentDecomposition>> allDecompositions = new ConcurrentHashMap<>();

        Consumer<String> computeMMPDecomp = (xidc) -> {
            StereoMolecule mi = ChemUtils.parseIDCode(xidc);
            List<FragmentDecomposition> decompositions_i =
                    FragmentDecompositionShredder.computeFragmentDecompositions(mi,idcodeToMolid.get(xidc),
                    max_fragment_size,max_relative_fragment_size,
                    min_extension_size,
                    max_splits);
            allDecompositions.put(xidc,decompositions_i);
        };

        int ncores = Runtime. getRuntime().availableProcessors();
        try {
            //Parallelizer.computeParallelBlocking(computeMMPDecomp,new ArrayList<>(sortedResults.keySet()),ncores);
            Parallelizer.computeParallelBlocking(computeMMPDecomp,new ArrayList<>(idcodeToMolid.keySet()),ncores);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        // create all MMPFragmentDecomposition objects, one per fragmentDecompId
        //List<MMPFragmentDecomposition> allFragmentDecompositions = new ArrayList<>();
        Map<String, MMPFragmentDecomposition> allFragmentDecompositions = new HashMap<>();
        for(String ki : allDecompositions.keySet()) {
            for(FragmentDecomposition fdi : allDecompositions.get(ki)) {
                FragmentDecompositionSynthon fdsi = new FragmentDecompositionSynthon(fdi);
                List<Long> assay_result_ids = sortedResults.get(idcodeToMolid.get(ki)).stream().map(ri -> ri.getId()).collect(Collectors.toList());
                MMPFragmentDecompositionImpl decomp_i = new MMPFragmentDecompositionImpl(fdsi,assay_id,assay_result_ids);
                allFragmentDecompositions.put(decomp_i.getDecompositionID(),decomp_i);
            }
        }
        return new ArrayList<>( allFragmentDecompositions.values() );
    }

}
