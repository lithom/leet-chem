package tech.molecules.leet.chem.dataimport;

import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.core.JsonProcessingException;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.CombinatoricsUtils;
import tech.molecules.leet.chem.LeetSerialization;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;
import tech.molecules.leet.chem.mutator.SimpleSynthonWithContext;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;
import tech.molecules.leet.chem.shredder.FragmentDecompositionShredder;
import tech.molecules.leet.io.CSVIterator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class FragmentDBCreator {

    private List<Pair<String,String>> molecules = new ArrayList<>();

    /**
     *
     * @param molecules left is id, right id idcode
     */
    public FragmentDBCreator(List<Pair<String,String>> molecules) {
        this.molecules = molecules;
    }

    public void init() {
        for(int zi=0;zi<molecules.size();zi++) {
            StereoMolecule mi = ChemUtils.parseIDCode(molecules.get(zi).getRight());
            processStructure(molecules.get(zi).getLeft(),mi);

            if(zi%200==0) {
                System.gc();
            }
        }
    }

    public void writeToCSV(String path) throws IOException {
        try(BufferedWriter out = new BufferedWriter(new FileWriter(path+ File.separator+"fragdb.csv"))) {
            List<String> header = new ArrayList<>();
            header.add("Fragment[idcode]");
            //header.add("Decomp[idcode]");
            header.add("bdir1[idcode]");
            header.add("count");
            out.write(String.join(",",header)+"\n");

            for(String bdir1 : this.decompositions_SortedByBidir1AndByCF.keySet()) {
                for(String cf : this.decompositions_SortedByBidir1AndByCF.get(bdir1).keySet() ) {
                    List<String> cols = new ArrayList<>();
                    cols.add(cf);
                    cols.add(bdir1);
                    cols.add(""+this.decompositions_SortedByBidir1AndByCF.get(bdir1).get(cf).size());
                    out.write(String.join(",",cols) + "\n");
                }
            }

            out.flush();
        }
    }

    private void processStructure(String id, StereoMolecule mi) {
        List<FragmentDecomposition> decompositions = FragmentDecompositionShredder.computeFragmentDecompositions(mi,id,18,0.4,3,3);
        for(FragmentDecomposition fdi : decompositions) {
            if(fdi.getInnerNeighborAtomicNos().stream().allMatch( ci -> ci==6 )) {
                addDecomposition(fdi);
            }
            //if(decompositions.size()>20){break;}
            //FragmentDecompositionSynthon fds = new FragmentDecompositionSynthon(fdi);
            //fds.
        }

    }


    /**
     * bidir1 -> ( cf -> ListOfDecompositions )
     */
    private Map<String,Map<String,List<String>>> decompositions_SortedByBidir1AndByCF = new HashMap<>();

    private synchronized void addDecomposition(FragmentDecomposition decomp) {
        String bidir1_idc = decomp.getBidirectionalConnectorProximalRegion(1).getIDCode();
        String cf_idc = decomp.getCentralFrag().getIDCode();

        if(!this.decompositions_SortedByBidir1AndByCF.containsKey(bidir1_idc)) {this.decompositions_SortedByBidir1AndByCF.put(bidir1_idc, new HashMap<>());}
        if(!this.decompositions_SortedByBidir1AndByCF.get(bidir1_idc).containsKey(cf_idc)) {this.decompositions_SortedByBidir1AndByCF.get(bidir1_idc).put(cf_idc, new ArrayList<>());}

        String str_decomp = null;
        try {
            str_decomp = LeetSerialization.OBJECT_MAPPER.writeValueAsString(decomp);
        } catch (JsonProcessingException e) {
            System.out.println("[ERROR] problem with serializatino of decomposition");
        }
        //this.decompositions_SortedByBidir1AndByCF.get(bidir1_idc).get(cf_idc).add(decomp);
        this.decompositions_SortedByBidir1AndByCF.get(bidir1_idc).get(cf_idc).add(str_decomp);
    }


    public static void main(String args[]) {
        List<Integer> fields = new ArrayList<>(); fields.add(0); fields.add(1);
        CSVIterator iter = null;
        try {
            iter = new CSVIterator("C:\\Temp\\leet\\chembl_structures_short.csv",true, fields);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int cnt = 0;
        List<Pair<String,String>> molecules = new ArrayList<>();
        while( iter.hasNext() ) { //&& cnt < 15000) {
            List<String> di = iter.next();
            molecules.add(Pair.of(di.get(0),di.get(1)));
            cnt++;
        }

        FragmentDBCreator fdbc = new FragmentDBCreator(molecules);
        fdbc.init();

        try {
            fdbc.writeToCSV("C:\\Temp\\leet");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Map<String,List<Pair<String,Integer>>> loadFragments() {
        Map<String,List<Pair<String,Integer>>> fragdb = new HashMap<>();
        String file = "C:\\Temp\\leet\\fragdb.csv";
        List<Integer> cols = new ArrayList<>();
        CSVIterator it = null;
        try {
            it = new CSVIterator(file,true, CombinatoricsUtils.intSeq(3));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        while(it.hasNext()) {
            try {
                List<String> line = it.next();
                if (!fragdb.containsKey(line.get(1))) {
                    fragdb.put(line.get(1), new ArrayList<>());
                }
                fragdb.get(line.get(1)).add(Pair.of(line.get(0), Integer.parseInt(line.get(2))));
            }
            catch(Exception ex) {
                System.out.println("[WARN] skip line, error parsing..");
            }
        }
        return fragdb;
    }

    /**
     *
     * @return serialized simple synthon objects with count
     */
    public static List<Pair<String,Integer>> loadFragments2() {
        //Map<String,List<Pair<String,Integer>>> fragdb = new HashMap<>();
        List<Pair<String,Integer>> fragmentdb = new ArrayList<>();
        String file = "C:\\Temp\\leet\\fragdb.csv";
        List<Integer> cols = new ArrayList<>();
        CSVIterator it = null;
        try {
            it = new CSVIterator(file,true, CombinatoricsUtils.intSeq(3));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        while(it.hasNext()) {
            try {
                List<String> line = it.next();
                StereoMolecule synthon = ChemUtils.parseIDCode(line.get(0));
                StereoMolecule context = ChemUtils.parseIDCode(line.get(1));
                List<SimpleSynthonWithContext> synthons = SimpleSynthonWithContext.createAllPossibleFromSynthonAndBidirectionalContext(synthon,context);

                fragmentdb.addAll(synthons.stream().map(si -> {
                    try {
                        return Pair.of( LeetSerialization.OBJECT_MAPPER.writeValueAsString(si) ,Integer.parseInt(line.get(2)));
                    } catch (JsonProcessingException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList()));
            }
            catch(Exception ex) {
                System.out.println("[WARN] skip line, error parsing..");
                ex.printStackTrace();
            }
        }
        return fragmentdb;
    }

}
