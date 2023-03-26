package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.conf.Conformer;
import com.actelion.research.chem.conf.ConformerSet;
import com.actelion.research.chem.conf.ConformerSetGenerator;
import org.apache.commons.lang3.tuple.Pair;
import org.openmolecules.chem.conf.gen.ConformerGenerator;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.IOUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class FlexoDataMiner {

//    public static class FlexoPoint {
//        //public final int dist_a,dist_b;
//        public final int element_a, element_b;
//        public final int ringsize;
//        public final boolean aromatic;
//        public final int neighbors_a;
//        public final int neighbors_b;
//
//
//    }

    public static class FlexoHistogram {
        private List<Double> values;

        public FlexoHistogram(List<Double> values) {
            this.values = values;
        }

        public int[] getHistogram( int bins, double min, double max, double smoothing) {
            Random r = new Random(567);
            double[] values_smoothed = values.stream().mapToDouble(di -> di*(1+(smoothing * r.nextGaussian()))).toArray();
            int[] h = new int[bins];
            for(int zi=0;zi<bins;zi++) {
                double xa = min + zi*(max-min)/(bins);
                double xb = min + (zi+1)*(max-min)/(bins);
                h[zi] = (int) Arrays.stream(values_smoothed).filter( xi -> xi>=xa && xi < xb ).count();
            }
            return h;
        }

    }


    public static void main(String args[]) {
        //List<String> mols = ChemUtils.loadTestMolecules_1795FromDrugCentral().stream().map(xi -> xi.getIDCode()).collect(Collectors.toList());
        // We reverse to start with the most drug unlike molecules..
        //Collections.reverse(mols);

        List<String> mols = IOUtils.extractSpecialColumnFromDWAR("C:\\datasets\\chembl_basic_dataset_for_fastflex.dwar","Structure");

        Random r = new Random(123);
        Collections.shuffle(mols,r);
        List<String> mols_A = mols.subList(0,4000);//mols;//mols.subList(0,50000);

        FlexoDataMiner miner = new FlexoDataMiner();
        List<FlexoDataset.FlexoDatapoint> fdp_list = miner.mineData(mols_A);

        FlexoDataset fd = new FlexoDataset(fdp_list);
        try {
            fd.createPyTorchInputData_Standard256(fd,"flexo_data_small.csv","flexo_encoder_data_small.csv");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private int samplesPerMolecule = 100;
    private int maxPathLength      = 32;

    /**    /**
     *
     * Computation of Macrocyclic score:
     * For a pair A and B at distance d, we search for the "longest" path that connects them.
     * We do this as follows. We pick the vertex with lower degree, and then we
     *
     *
     * @param a
     * @param b
     * @param computeMacrocyclicScore
     * @return left is list of path spheres, right is macrocyclic score (if computed)
     */
    public static Pair<List<String>,Double> createPathSpheres(StereoMolecule mi_a, int a, int b, boolean computeMacrocyclicScore) {
        mi_a.ensureHelperArrays(Molecule.cHelperCIP);

        StereoMolecule mi = new StereoMolecule(mi_a);
        mi.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<mi.getAtoms();zi++) {
            if( mi.getAtomicNo(zi) == 17 || mi.getAtomicNo(zi) == 35 || mi.getAtomicNo(zi) == 53 ) {
                mi.setAtomicNo(zi,9);
            }
        }

        List<String> out =new ArrayList<>();
        int[] path_data = new int[200];
        int length = mi.getPath(path_data,a,b,60,null);

        if(length<0) {return null;}
        // for first and last, cut off all parts that are not connected to the "inner" part:
        // nah.. cutoff everything that is not in path direction..
        if(false) {
            mi.setAtomCustomLabel(path_data[0], "XA");
            mi.setAtomCustomLabel(path_data[length], "XB");
            System.out.println(new Canonizer(mi, Canonizer.ENCODE_ATOM_CUSTOM_LABELS).getIDCode());
        }
        if(true) {
            // condition for being in: has path to first inner and last inner atom.
            boolean[] remove = new boolean[200];
            int idx_first_inner = path_data[1];
            int idx_last_inner  = path_data[Math.max(1,path_data.length-2)];
            for(int zi=0;zi<mi.getConnAtoms(a);zi++) {
                //if( mi.getConnAtom(a,zi) != idx_first_inner && mi.getConnAtom(a,zi) != idx_last_inner ) { remove[mi.getConnAtom(a,zi)] = true; }
                int fzi = zi;
                if(!Arrays.stream(path_data).anyMatch( xi -> xi == mi.getConnAtom(a,fzi) )) {
                    remove[mi.getConnAtom(a,fzi)] = true;
                }
            }
            for(int zi=0;zi<mi.getConnAtoms(b);zi++) {
                //if( mi.getConnAtom(b,zi) != idx_first_inner && mi.getConnAtom(b,zi) != idx_last_inner ) { remove[mi.getConnAtom(b,zi)] = true; }
                int fzi = zi;
                if(!Arrays.stream(path_data).anyMatch( xi -> xi == mi.getConnAtom(b,fzi) )) {
                    remove[mi.getConnAtom(b,fzi)] = true;
                }
            }

            int[] old_to_new = mi.deleteAtoms(remove);
            if(old_to_new!=null) {
                for (int zi = 0; zi <= length; zi++) {
                    path_data[zi] = old_to_new[path_data[zi]];
                }
                mi.ensureHelperArrays(Molecule.cHelperCIP);
            }
            if(false) {
                System.out.println(mi.getIDCode());
            }
        }

        for(int zi=0;zi<length;zi++) {

            int old_before = -1;
            int old_after  = -1;
            if(zi>0){
                old_before = mi.getAtomicNo(path_data[zi-1]);
                mi.setAtomicNo(path_data[zi-1],92);
            }
            if(zi<length-1){
                old_after = mi.getAtomicNo(path_data[zi+1]);
                mi.setAtomicNo(path_data[zi+1],93);
            }
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            StereoMolecule fi = ChemUtils.createProximalFragment(mi, Collections.singletonList(path_data[zi]),
                    1,false,null);


            out.add(fi.getIDCode());

            fi.ensureHelperArrays(Molecule.cHelperCIP);
            // count special elements:
            int transurans = 0;
            for(int zx=0;zx<fi.getAtoms();zx++) { if(fi.getAtomicNo(zx)>=90){transurans++;} }
            if(transurans ==0 ) {
                System.out.println("mkay.. 0");
            }

            if(old_before>=0) { mi.setAtomicNo(path_data[zi-1],old_before); }
            if(old_after>=0) { mi.setAtomicNo(path_data[zi+1],old_after); }

            mi.ensureHelperArrays(Molecule.cHelperCIP);
        }

        Double mcScore = Double.NaN;
        if(false) { // TODO.. hmm..
            mcScore = 0.0d;
            // TODO: implement..
            int ca=-1;
            int cb=-1;
            if( mi.getConnAtoms(a) < mi.getConnAtoms(b)) {
                ca = a; cb = b;
            }
            else {
                cb = a; cb = a;
            }
            boolean[] neglected_bonds = new boolean[mi.getConnAtoms(ca)];
            //int max_shortest_path_length =
            for(int zi=0;zi<mi.getConnAtoms(ca);zi++) {
                int[] pathAtoms = new int[60];
                int li = mi.getPath(pathAtoms,ca,cb,60,neglected_bonds);
                neglected_bonds[ mi.getBond(ca,pathAtoms[1]) ] = true;
                if(zi>0) {

                }
            }
        }

        return Pair.of(out,mcScore);
    }


    public static FlexoHistogram createFlexoHistogram(ConformerSet confSet, int a, int b, int minSamples) throws Exception {
        Random r = new Random(567);
        List<Double> values = new ArrayList<>();
        int multiplicator = 1;
        if(confSet.size()<minSamples) {
            double fmult = (1.0*minSamples) / confSet.size();
            multiplicator = Math.max( (int)fmult,1);
        }
        for(Conformer ci : confSet) {
            for(int zi=0;zi<multiplicator;zi++) {
                values.add(ci.getCoordinates(a).distance(ci.getCoordinates(b)));
            }
        }
        return new FlexoHistogram(values);
    }

    public List<FlexoDataset.FlexoDatapoint> mineData(List<String> input) {

        List<FlexoDataset.FlexoDatapoint> dataset = new ArrayList<>();

        Map<String,Integer> all_paths = new HashMap<>();
        double max_dist = 0;

        int cnt = 0;

        Random r = new Random(1234);
        ConformerSetGenerator csg = new ConformerSetGenerator();

        BufferedWriter out = null;
        try {
            out = new BufferedWriter(new FileWriter( "mined_flexo_100bins.csv" ));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        for(int zi=0;zi<input.size();zi++) {
            StereoMolecule mi = ChemUtils.parseIDCode(input.get(zi));
            mi.ensureHelperArrays(Molecule.cHelperCIP);

            if(mi.getAtoms()<10) {continue;}

            // create 3d data:
            ConformerSet confSet = new ConformerSet();
            confSet = csg.generateConformerSet(mi);
            // little hack to make the histogram smoothing work..
//            if(confSet.size()<40) {
//                int multiplicator = (int) (40.0 / confSet.size());
//                if(multiplicator>1) {
//                    ConformerSet confSet_extended = new ConformerSet();
//                    for(Conformer ci : confSet) {
//                        ci.setX( 0 , ci.getX(0)+ 0.001*r.nextDouble() );
//                        for(int zj=0;zj<multiplicator;zj++) { confSet_extended.add(ci); }
//                    }
//                    confSet = confSet_extended;
//                }
//            }

            for(int zj=0;zj< Math.min( ((mi.getAtoms()-8)*(mi.getAtoms()-8)) , samplesPerMolecule ) ;zj++) {

                boolean ok = false;
                int xa = 0;
                int xb = 0;
                while(!ok) {
                    xa = r.nextInt(mi.getAtoms());
                    xb = r.nextInt(mi.getAtoms());
                    if(ChemUtils.checkNeighborAtoms(mi,xa,xb)) {continue;}
                    ok = xa!=xb;

                }

                // create path data:+
                Pair<List<String>,Double> path_and_mcScore = createPathSpheres(mi,xa,xb,true);
                if(path_and_mcScore==null) {
                    System.out.println("null..");
                    continue;
                }

                List<String> path = path_and_mcScore.getLeft();
                for(String pi : path) {
                    if(!all_paths.containsKey(pi)) { all_paths.put(pi,1); }
                    else{all_paths.put(pi,all_paths.get(pi)+1);}
                }

                if(path.size()>maxPathLength) {
                    System.out.println("[WARN] Max path length exceeded -> skip");
                    continue;
                }


                //System.out.println("s="+all_paths.size());
                cnt++;
                if(cnt%10000==0) {
                    System.out.println("Unique frags: "+all_paths.size());
                    System.out.println("Most common frags:");
                    System.out.println("Frag[idcode]\tCount");
                    List<Pair<Integer,String>> pi = all_paths.entrySet().stream().map( xi -> Pair.of(xi.getValue(),xi.getKey()) ).collect(Collectors.toList());
                    pi.sort( (x,y) -> -Integer.compare(x.getKey(),y.getKey()) );
                    for(int zx=0;zx<pi.size();zx++) {
                        //System.out.println(pi.get(zx).getRight()+"\t"+pi.get(zx).getLeft());
                    }
                }

                FlexoHistogram hist = null;
                try {
                    hist = createFlexoHistogram(confSet, xa, xb, 100);
                }
                catch(Exception ex) {
                    continue;
                }
                int[] counts = hist.getHistogram(100,0,50,0.05);
                max_dist = Math.max(max_dist, hist.values.stream().mapToDouble(di -> di).max().getAsDouble());
                System.out.println("max_dist="+max_dist);
                System.out.println("mkay..");

                List<String> all_path_strings= new ArrayList<>();
                for(int zp=0;zp<maxPathLength;zp++) {
                    if(zp<path.size()) {all_path_strings.add(path.get(zp));}
                    else{all_path_strings.add("");}
                }

                //double[] hist_normalized = new double[counts.length];
                //double sum_hist          = Arrays.stream(counts).sum();
                //for(int zk=0;zk<hist_normalized.length;zk++) { hist_normalized[zk] = 1.0*counts[zi]/; }

                FlexoDataset.FlexoDatapoint pi = new FlexoDataset.FlexoDatapoint(all_path_strings,Double.NaN,counts);
                dataset.add(pi);

                List<String> all_histogram_strings = Arrays.stream(counts).mapToObj( xi -> ""+xi ).collect(Collectors.toList());
                try {
                    out.write(String.join(",", all_path_strings) + "," + String.join(",",all_histogram_strings));
                    out.write("\n");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        return dataset;
    }

}
