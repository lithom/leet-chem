package tech.molecules.leet.chem.shredder;

import com.actelion.research.calc.combinatorics.CombinationGenerator;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;

import java.util.*;
import java.util.stream.Collectors;

public class FragmentDecompositionShredder {


    /**
     *
     *
     * @param m
     * @param min_splits
     * @param max_splits
     * @param atom_pairs_to_cut atomic number pairs to consider. NOTE!! Must be sorted ascending
     * @param prevent_adjacent_cuts
     * @param cut_ring_bonds
     * @param cut_single_bonds
     * @param cut_double_bonds
     * @param cut_triple_bonds
     * @return
     */
    public static List<SynthonShredder.SplitResult> computeSplitResults(StereoMolecule m,
                                                                        int min_splits, int max_splits,
                                                                        int max_fragments,
                                                                        Set<List<Integer>> atom_pairs_to_cut,
                                                                        boolean prevent_adjacent_cuts,
                                                                        boolean cut_ring_bonds,
                                                                        boolean cut_single_bonds,
                                                                        boolean cut_double_bonds,
                                                                        boolean cut_triple_bonds) {

        m.ensureHelperArrays(StereoMolecule.cHelperCIP);

        // first: filter edges that can be considered for cuts:
        List<Integer> bonds_for_cuts = new ArrayList<>();
        for(int zi=0;zi<m.getBonds();zi++) {
             List bi = new ArrayList();
             int aba = m.getAtomicNo(m.getBondAtom(0,zi));
             int abb = m.getAtomicNo(m.getBondAtom(1,zi));
             if(aba<abb) {bi.add(aba);bi.add(abb);}
             else{ bi.add( abb ); bi.add(aba);}

             if(!atom_pairs_to_cut.contains(bi)) {
                 continue;
             }

             // check bond type is ok:
            if(!cut_ring_bonds) {
                if( m.isRingBond(zi) ) {
                    continue;
                }
            }
            int bond_order = m.getBondOrder(zi);
            if(!cut_single_bonds && bond_order==1) {
                continue;
            }
            if(!cut_double_bonds && bond_order==2) {
                continue;
            }
            if(!cut_triple_bonds && bond_order==3) {
                continue;
            }

            // ok, consider this bond
            bonds_for_cuts.add(zi);
        }

        List<SynthonShredder.SplitResult> all_split_results = new ArrayList<>();
        for(int splits = min_splits; splits<=max_splits;splits++) {
            if(bonds_for_cuts.size()<splits) {continue;}
            List<int[]> all_cuts = CombinationGenerator.getAllOutOf(bonds_for_cuts.size(), splits);

            List<int[]> all_edge_cuts = all_cuts.parallelStream().map(aci ->
                    Arrays.stream(aci).map(ei -> bonds_for_cuts.get(ei) ).toArray() ).collect(Collectors.toList());

            if(prevent_adjacent_cuts) {
                all_edge_cuts = all_edge_cuts.parallelStream().filter(ei -> !ChemUtils.checkForAdjacentBonds(m,ei) ).collect(Collectors.toList());
            }

            List<SynthonShredder.SplitResult> split_results = all_edge_cuts.parallelStream().map(xi ->
                        SynthonShredder.trySplit(m,xi,max_fragments) ).filter(si -> si!=null ).collect(Collectors.toList());
            all_split_results.addAll(split_results);
        }

        return all_split_results;
    }

    /**
     *
     * @param m
     * @param moleculeId
     * @param max_fragment_size
     * @param max_relative_fragment_size
     * @param min_extension_size minimum number of hacs in every non-central fragment
     * @param max_splits
     * @return
     */
    public static List<FragmentDecomposition> computeFragmentDecompositions(StereoMolecule m,
                                                                            String moleculeId,
                                                                            int max_fragment_size, double max_relative_fragment_size,
                                                                            int min_extension_size,
                                                                            int max_splits) {

        Set<List<Integer>> edgesToConsider = createIntPairList( new int[][]{ {6,6} , {6,7} , {6,8} , {6,9} , {6,16} , {6,17} , {6,35} });

        List<SynthonShredder.SplitResult> splits = computeSplitResults(m,1,max_splits,max_splits+1,edgesToConsider,true,false,true,false,false);
        List<FragmentDecomposition> decompositions = new ArrayList<>();

        for(int zi=0;zi<splits.size();zi++) {
            SynthonShredder.SplitResult sri = splits.get(zi);
            for(int zf=0;zf<sri.fragments.length;zf++) {
                if( FragmentDecomposition.checkIsFragmentDecomposition(sri,zf) ) {
                    double total_size = Arrays.stream( sri.fragments ).mapToInt( fi -> SynthonUtils.countNonConnectorAtoms(fi) ).sum();
                    double size_i = SynthonUtils.countNonConnectorAtoms(sri.fragments[zf]);
                    double rel_frag_size = size_i/total_size;
                    if(size_i<=max_fragment_size && rel_frag_size < max_relative_fragment_size) {
                        FragmentDecomposition fdi = new FragmentDecomposition(moleculeId,sri,zf);
                        if( fdi.getMinExtensionSize() >= min_extension_size) {
                            decompositions.add(fdi);
                        }
                    }
                }
            }
        }

        return decompositions;
    }


    public static Set<List<Integer>> createIntPairList(int[][] pairs) {
        Set<List<Integer>> set = new HashSet<>();
        for(int zi=0;zi<pairs.length;zi++) {
            List<Integer> li = new ArrayList<>(); li.add(pairs[zi][0]); li.add(pairs[zi][1]);
            set.add(li);
        }
        return set;
    }


    public static void main(String args[]) {
        String ma = "";//;
        StereoMolecule mi = ChemUtils.parseIDCode(ma);

        List<FragmentDecomposition> decompositions = computeFragmentDecompositions(mi,"test",16,0.4,3,4);

        List<StereoMolecule> mols   = decompositions.stream().map( di -> di.getFragmentsWithHighlighting() ).collect(Collectors.toList());
        List<StereoMolecule> mols_1 = decompositions.stream().map( di -> di.getBidirectionalConnectorProximalRegion(1) ).collect(Collectors.toList());
        List<StereoMolecule> mols_2 = decompositions.stream().map( di -> di.getBidirectionalConnectorProximalRegion(2) ).collect(Collectors.toList());
        List<StereoMolecule> mols_3 = decompositions.stream().map( di -> new FragmentDecompositionSynthon(di).getContextBidirectirectional(3,2) ).collect(Collectors.toList());
        List<StereoMolecule> mols_4 = decompositions.stream().map( di -> new FragmentDecompositionSynthon(di).getContextBidirectirectional(1,1) ).collect(Collectors.toList());

        ChemUtils.DebugOutput.plotMolecules("test",mols.toArray(new StereoMolecule[0]),8,8);
        ChemUtils.DebugOutput.plotMolecules("test1",mols_1.toArray(new StereoMolecule[0]),8,8);
        ChemUtils.DebugOutput.plotMolecules("test2",mols_2.toArray(new StereoMolecule[0]),8,8);
        ChemUtils.DebugOutput.plotMolecules("test3",mols_3.toArray(new StereoMolecule[0]),8,8);
        ChemUtils.DebugOutput.plotMolecules("test4",mols_4.toArray(new StereoMolecule[0]),8,8);
        System.out.println("mkay");
    }

}
