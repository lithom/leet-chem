package tech.molecules.leet.chem.shredder;

import com.actelion.research.chem.*;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import com.actelion.research.chem.shredder.FragmentGenerator;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.QueryFeatureUtils;

import java.util.*;
import java.util.stream.Collectors;

public class ConnectedPairSampler {

    private StereoMolecule M;

    public ConnectedPairSampler(StereoMolecule m) {
        this.M = new StereoMolecule(m);
        this.M.ensureHelperArrays(Molecule.cHelperCIP);
    }

    /**
     * Samples fragments that are non-overlapping
     */
    public List<ConnectedPairOfSubgraphs> computePairs(int min_size, int max_size, int min_distance, int max_distance) {
        FragmentGenerator fg = new FragmentGenerator(this.M,min_size,max_size);
        fg.computeFragments();

        List<BitSet> frags_all = fg.getFragments();

        // filter frags according to size:
        //List<BitSet> frags = new ArrayList<>( frags_all.stream().filter(
        //        fi -> fi.cardinality() >= min_size && fi.cardinality() <= max_size ).collect(Collectors.toList()) );

        List<ConnectedPairOfSubgraphs> cps_all = computePairsOfSubgraphs(this.M,frags_all);

        // filter cps:
        List<ConnectedPairOfSubgraphs> cps = new ArrayList<>( cps_all.stream().filter(
                pi -> pi.dist >= min_distance && pi.dist <= max_distance ).collect(Collectors.toList()));

        return cps;
    }

    public List<ConnectedPairOfSubgraphs> computePairsWithClusteredSubgraphs(int min_size, int max_size, int min_distance, int max_distance, int clusters) {
        List<Pair<BitSet,StereoMolecule>> cfrags = computeRepresentativeSubgraphs(this.M, min_size, max_size, clusters);

        List<ConnectedPairOfSubgraphs> cps_all = computePairsOfSubgraphs(this.M,cfrags.stream().map( pi -> pi.getLeft() ).collect(Collectors.toList()) );

        // filter cps:
        List<ConnectedPairOfSubgraphs> cps = new ArrayList<>( cps_all.stream().filter(
                pi -> pi.dist >= min_distance && pi.dist <= max_distance ).collect(Collectors.toList()));

        return cps;
    }

    public static List<Pair<BitSet,StereoMolecule>> computeRepresentativeSubgraphs(StereoMolecule m, int min_size, int max_size, int clusters) {
        DiversitySelector selector = new DiversitySelector();
        selector.initializeExistingSet(512);

        DescriptorHandlerLongFFP512 dh = (DescriptorHandlerLongFFP512) DescriptorHandlerLongFFP512.getDefaultInstance();

        List<Pair<BitSet,StereoMolecule>> result = new ArrayList<>();

        FragmentGenerator fg = new FragmentGenerator(m,min_size,max_size);
        List<BitSet> frags = fg.getFragments();
        Map<BitSet,StereoMolecule> fragMolecules = new HashMap<>();

        // process synthons and compute descriptors:
        long[][] descriptors = new long[frags.size()][];

        for(int zi=0;zi<frags.size();zi++) {
            StereoMolecule mi = new StereoMolecule();
            m.copyMoleculeByAtoms(mi,ChemUtils.toBooleanArray(frags.get(zi)),true,null);
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            fragMolecules.put(frags.get(zi),mi);

            descriptors[zi] = dh.createDescriptor(mi);
        }

        int[] selected = selector.select( descriptors , clusters );

        for(int si : selected) {
            result.add(Pair.of( frags.get(si) , fragMolecules.get(frags.get(si))) );
            if(true) {
                System.out.println(fragMolecules.get(frags.get(si)).getIDCode());
            }
        }
        return result;
    }

    public static class ConnectedPairOfSubgraphs {
        public final StereoMolecule m;
        public final BitSet frag_a;
        public final BitSet frag_b;
        public final int anchor_a;
        public final int anchor_b;
        public final int dist;

        public ConnectedPairOfSubgraphs(StereoMolecule m, BitSet frag_a, BitSet frag_b, int anchor_a, int anchor_b, int dist) {
            this.m = m;
            this.frag_a = frag_a;
            this.frag_b = frag_b;
            this.anchor_a = anchor_a;
            this.anchor_b = anchor_b;
            this.dist = dist;
        }

        public StereoMolecule getPairAsMoleculeWithBridgeQF(int minLength, int maxLength) {
            StereoMolecule mnew = new StereoMolecule();

            StereoMolecule fa = new StereoMolecule();
            StereoMolecule fb = new StereoMolecule();
            int map_fa[] = new int[m.getAtoms()];
            int map_fb[] = new int[m.getAtoms()];
            this.m.copyMoleculeByAtoms(fa, ChemUtils.toBooleanArray(frag_a,m.getAtoms()) , true, map_fa);
            this.m.copyMoleculeByAtoms(fb, ChemUtils.toBooleanArray(frag_b,m.getAtoms()) , true, map_fb);

            fa.ensureHelperArrays(Molecule.cHelperCIP);
            fb.ensureHelperArrays(Molecule.cHelperCIP);

            int map_2_fa[] = new int[fa.getAtoms()];
            int map_2_fb[] = new int[fb.getAtoms()];

            mnew.addFragment(fa,0,map_2_fa);
            mnew.addFragment(fb,0,map_2_fb);

            // now add bridged bond in between:
            int atom_a = map_2_fa[ map_fa [ anchor_a ] ];
            int atom_b = map_2_fb[ map_fb [ anchor_b ] ];

            QueryFeatureUtils.addBridgeBond(mnew,atom_a,atom_b,minLength,maxLength);
            return mnew;
        }
    }

    public static List<ConnectedPairOfSubgraphs> computePairsOfSubgraphs(StereoMolecule m, List<BitSet> frags) {
        List<ConnectedPairOfSubgraphs> cps = new ArrayList<>();
        for(int zi=0;zi<frags.size()-1;zi++) {
            for(int zj=zi;zj<frags.size();zj++) {
                BitSet fa = (BitSet) frags.get(zi).clone();
                BitSet fb = (BitSet) frags.get(zj).clone();
                // check overlap
                if( fa.intersects(fb) ) {continue;}

                // compute distance
                int min_dist = 1000000;
                int anchor_a = -1; int anchor_b = -1;
                for(int zai : fa.stream().toArray()) {
                    for(int zbi : fb.stream().toArray()) {
                        int pdi = m.getPathLength(zai,zbi); // TODO: probably add same frag atoms as "dont use" to speed up?..
                        if(pdi<min_dist) {
                            min_dist = pdi; anchor_a = zai; anchor_b = zbi;
                        }
                    }
                }
                cps.add(new ConnectedPairOfSubgraphs(m,fa,fb,anchor_a,anchor_b,min_dist));
            }
        }
        return cps;
    }


    public static void main(String args[]) {
        String mol_a = "O=C(Nc1ncnc2nc[nH]c12)c1cccc(-c2cccc(-c3cn(CP(=O)(O)O)cn3)c2)c1"; // from chembl
        String mol_b = "CC(C)(C)c(cc1)ccc1C(N[C@@H](CCCCN1Cc2ccccc2)C1=O)=O"; // from Idorsia
        StereoMolecule ma = ChemUtils.parseSmiles(mol_b);

        ConnectedPairSampler cps = new ConnectedPairSampler(ma);

        //List<ConnectedPairOfSubgraphs> pairs = cps.computePairs(10, 10, 3, 7);
        List<ConnectedPairOfSubgraphs> pairs = cps.computePairsWithClusteredSubgraphs(9, 9, 2, 5, 20);

        System.out.println("mkay..");

        //Collections.shuffle(pairs);
        for(int zi=0;zi< Math.min( 1000 , pairs.size() ) ;zi++) {
            int di = pairs.get(zi).dist;
            StereoMolecule mi = pairs.get(zi).getPairAsMoleculeWithBridgeQF( di-1 , di+1 );
            System.out.println(mi.getIDCode());
        }

    }

}
