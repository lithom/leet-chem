package tech.molecules.leet.chem.shredder;


import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.coords.CoordinateInventor;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.CombinatoricsUtils;
import tech.molecules.leet.chem.mutator.SimpleSynthonWithContext;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a SplitResult that contains a central fragment,
 * in the sense that the central fragment contains half of
 * all connectors, i.e. is connected to every other
 * fragment in teh split result
 *
 *
 */
public class FragmentDecomposition implements Serializable {

    @JsonPropertyDescription("molecule id")
    @JsonProperty("id")
    private String molid;

    @JsonPropertyDescription("split result")
    @JsonProperty("sr")
    private SynthonShredder.SplitResult splitResult;

    @JsonPropertyDescription("central fragment")
    @JsonProperty("cf")
    private int centralFrag;

    public FragmentDecomposition(String molid,SynthonShredder.SplitResult si, int central_frag) {
        this.molid = molid;
        this.splitResult = si;
        this.centralFrag = central_frag;

        init();
    }

    private void init() {

    }

    public SynthonShredder.SplitResult getSplitResult() {
        return this.splitResult;
    }

    public int getNumberOfConnectors() {
        return this.splitResult.connector_positions.get(0).length;
    }


    /**
     * Checks if the central frag is connected to all other fragments.
     *
     * @return
     */
    public boolean isFragmentDecomposition() { return checkIsFragmentDecomposition(this.splitResult,this.centralFrag); }

    /**
     * Returns the size of the smallest non central fragment (without connector atom)
     * @return
     */
    public int getMinExtensionSize() {
        List<Integer> to_check = CombinatoricsUtils.intSeq(this.splitResult.fragments.length, Collections.singletonList(this.centralFrag));
        return to_check.stream().mapToInt( ci -> splitResult.fragments[ci].getAtoms()-1 ).min().getAsInt();
    }

    /**
     *
     * @return
     */
    public List<Integer> getInnerNeighborAtomicNos() {
        StereoMolecule cf = this.getCentralFrag();
        BitSet bs = SynthonUtils.findConnectorAtoms( cf );
        List<Integer> nbs = new ArrayList<>();
        for( int ci : bs.stream().toArray()) {
            nbs.add(  cf.getAtomicNo( cf.getConnAtom(ci,0) ) );
        }
        return nbs;
    }


    public StereoMolecule getCentralFrag() {
        return this.splitResult.fragments[this.centralFrag];
    }

    public static boolean checkIsFragmentDecomposition(SynthonShredder.SplitResult si, int central_frag) {
        return Arrays.stream( si.connector_positions.get(central_frag) ).allMatch( pi -> pi>= 0 );
    }

    public SimpleSynthonWithContext toSimpleSynthonWithContext() {
        StereoMolecule synthon = new StereoMolecule();
        this.getCentralFrag().copyMolecule(synthon);
        int[][] cp_pair_positions = new int[this.getNumberOfConnectors()][];
        List<int[]> atom_maps = new ArrayList<>();
        StereoMolecule combined = this.createCombinedFragmentsMoleculeWithLinkerConnectors(cp_pair_positions,true,atom_maps);
        //int[] map_old_to_new = new int[combined.getAtoms()];
        //StereoMolecule cut = SynthonUtils.cutBidirectionalContext(combined,cp_pair_positions,1,1,map_old_to_new);


        int[] map_old_to_new_outer = new int[combined.getAtoms()];
        int[] map_old_to_new_inner = new int[combined.getAtoms()];
        StereoMolecule cut_outer = SynthonUtils.cutBidirectionalContext(combined,cp_pair_positions,0,1,map_old_to_new_outer);
        StereoMolecule cut_inner = SynthonUtils.cutBidirectionalContext(combined,cp_pair_positions,100000,0,map_old_to_new_inner);

        // remove "wrong" connis in outer / inner:
        List<Integer> outer_toRemove = ChemUtils.toIntList( ChemUtils.findAtomsWithAtomicNo(cut_outer,92) );
        List<Integer> inner_toRemove = ChemUtils.toIntList( ChemUtils.findAtomsWithAtomicNo(cut_inner,93) );
        for(int ori : outer_toRemove) {cut_outer.markAtomForDeletion(ori);}
        for(int iri : inner_toRemove) {cut_inner.markAtomForDeletion(iri);}
        int[] outer_amap = cut_outer.deleteMarkedAtomsAndBonds();
        int[] inner_amap = cut_inner.deleteMarkedAtomsAndBonds();

        cut_inner.ensureHelperArrays(Molecule.cHelperCIP);
        // then: for outer: change 93 connectors to 92 for the simple synthon..
        cut_outer.ensureHelperArrays(Molecule.cHelperNeighbours);
        for(int ci : ChemUtils.findAtomsWithAtomicNo(cut_outer,93).stream().toArray()){ cut_outer.setAtomicNo(ci,92); }
        cut_outer.ensureHelperArrays(Molecule.cHelperCIP);

        int[][] cp_pair_positions_2 = new int[this.getNumberOfConnectors()][];
        for(int zi=0;zi<cp_pair_positions.length;zi++) {
            cp_pair_positions_2[zi] = new int[]{ inner_amap[map_old_to_new_inner[cp_pair_positions[zi][0]]] , outer_amap[map_old_to_new_outer[cp_pair_positions[zi][1]]] };
        }

        return new SimpleSynthonWithContext(cut_inner,cut_outer,cp_pair_positions_2);
//        int[][] cp_pair_positions_2 = new int[this.getNumberOfConnectors()][];
//        for(int zi=0;zi<cp_pair_positions.length;zi++) {
//            cp_pair_positions_2[zi] = new int[]{  , map_old_to_new[cp_pair_positions[zi][1]] };
//        }

        //return new SimpleSynthonWithContext(synthon,cut,cp_pair_positions_2);
    }

    /**
     *
     * @return connector pairs, i.e. [i][0] contains conni pos of central fragment of conni that is connected to
     * i'th non-central fragment, [i][1] contains the conni pos in the extension fragment.
     *
     *
     */
    public int[][] getConnectorPairs() {
        List<Integer> rfrags = CombinatoricsUtils.intSeq(0,this.splitResult.fragments.length,Collections.singletonList(this.centralFrag));
        int connectorPairs[][] = new int[rfrags.size()][];
        for(int zi=0;zi<rfrags.size();zi++) {
            // find connector that is in i-th extension frag:
            int cxi = -1;
            for(int zj=0;zj<this.splitResult.connector_positions.get(rfrags.get(zi)).length;zj++) {
                if( this.splitResult.connector_positions.get(rfrags.get(zi))[zj]>=0 ) {
                    cxi = zj; break;
                }
            }

            // find positions of i'th connector:
            // 1. central frag:
            int pos_cf = this.splitResult.connector_positions.get(this.centralFrag)[cxi];

            connectorPairs[zi] = new int[]{ pos_cf , this.splitResult.connector_positions.get(rfrags.get(zi))[cxi] };
            //this.splitResult.connector_positions
        }
        return connectorPairs;
    }

    public StereoMolecule createCombinedFragmentsMolecule() {
        StereoMolecule ma = new StereoMolecule();
        for(int zf=0;zf<splitResult.fragments.length;zf++) {
            StereoMolecule fi = splitResult.fragments[zf];
            StereoMolecule fi_copy = new StereoMolecule(fi);
            fi_copy.ensureHelperArrays(Molecule.cHelperCIP);
            int amap[] = new int[fi_copy.getAtoms()];
            ma.addFragment(fi_copy,0,amap);
            //if(zf==centralFrag) {map_highlighting=amap;}
            //ChemUtils.highlightBondsInBetweenAtoms();
        }
        ma.ensureHelperArrays(Molecule.cHelperCIP);
        return ma;
    }

    /**
     * @param cp_pair_positions if not null and long enough will contain the atom positions
     *                          of the pairs of connectors (first inner, second outer).
     * @param addConnectorConnectorBonds
     * @param atom_maps i'th entry will contain atom positions of i'th splitresult fragment
     *                  in the new molecule
     *
     * @return an assembled molecule, but there are the two connector atoms inbetween
     * all assembled fragments. The inner connectors are U connectors, the outer connectors
     * are Np connectors.
     *
     */
    public StereoMolecule createCombinedFragmentsMoleculeWithLinkerConnectors(int cp_pair_positions[][], boolean addConnectorConnectorBonds, List<int[]> atom_maps) {
        StereoMolecule ma = new StereoMolecule();
        List<int[]> amaps = new ArrayList<>();
        for(int zf=0;zf<splitResult.fragments.length;zf++) {
            StereoMolecule fi = splitResult.fragments[zf];
            StereoMolecule fi_copy = new StereoMolecule(fi);
            fi_copy.ensureHelperArrays(Molecule.cHelperCIP);
            int amap[] = new int[fi_copy.getAtoms()];
            ma.addFragment(fi_copy,0,amap);
            amaps.add(amap);
            atom_maps.add(amap);
            //if(zf==centralFrag) {cf_map = amap;}
            //ChemUtils.highlightBondsInBetweenAtoms();
        }
        //ma.ensureHelperArrays(Molecule.cHelperCIP);
        // now assemble all connectors of the central fragment:
        int num_connectors = splitResult.connector_positions.get(centralFrag).length;
        if(num_connectors>1) {
            //System.out.println("connis: "+num_connectors);
        }
        else {
            //System.out.println("single connector");
        }
        for(int zc=0;zc<num_connectors;zc++) {
            int cfpos = splitResult.connector_positions.get(centralFrag)[zc];
            if(cfpos>=0) {
                // now search the counterpart:
                for (int zo : CombinatoricsUtils.intSeq(splitResult.fragments.length, Collections.singletonList(centralFrag))) {
                    if (splitResult.connector_positions.get(zo)[zc] >= 0) {
                        // ok we found it, assemble :)
                        int cfposnew = amaps.get(centralFrag)[cfpos];
                        int oposnew = amaps.get(zo)[splitResult.connector_positions.get(zo)[zc]];

                        //System.out.println("[INFO] connect: "+cfposnew+" <-> "+oposnew);
                        if (cp_pair_positions != null && cp_pair_positions.length > zc) {
                            cp_pair_positions[zc] = new int[]{cfposnew, oposnew};
                        }

                        // i.e. connect them..
                        if (addConnectorConnectorBonds) {
                            ma.addBond(cfposnew, oposnew, Molecule.cBondTypeSingle);
                        }
                        ma.setAtomicNo(oposnew, 93);
                    }
                }
            }
        }
        ma.ensureHelperArrays(Molecule.cHelperCIP);

        return ma;
    }


    public StereoMolecule getBidirectionalConnectorProximalRegion(int region_size) {
        // not like this, instead we do this separately for every connector
        //return SynthonUtils.createConnectorProximalFragment(this.createCombinedFragmentsMoleculeWithLinkerConnectors(),region_size);
        int[][] cp_pair_positions = new int[this.splitResult.connector_positions.get(0).length][2];

        List<int[]> atom_maps = new ArrayList<>();
        StereoMolecule lc = this.createCombinedFragmentsMoleculeWithLinkerConnectors(cp_pair_positions,true, atom_maps);

        StereoMolecule all_prs = new StereoMolecule();
        all_prs.setFragment(true);

        // create fragments
        for(int zp=0;zp<cp_pair_positions.length;zp++) {
            List<Integer> seed_atoms = Arrays.stream(cp_pair_positions[zp]).boxed().collect(Collectors.toList());
            StereoMolecule mi_a = ChemUtils.createProximalFragment(lc,seed_atoms,region_size,true,null);
            all_prs.addFragment(mi_a,0,null);
        }
        all_prs.ensureHelperArrays(Molecule.cHelperCIP);
        return all_prs;
    }


    public StereoMolecule getFragmentsWithHighlighting() {
        StereoMolecule ma = new StereoMolecule();
        int map_highlighting[] = null;
        for(int zf=0;zf<splitResult.fragments.length;zf++) {
            StereoMolecule fi = splitResult.fragments[zf];
            StereoMolecule fi_copy = new StereoMolecule(fi);
            fi_copy.ensureHelperArrays(Molecule.cHelperCIP);
            int amap[] = new int[fi_copy.getAtoms()];
            ma.addFragment(fi_copy,0,amap);
            if(zf==centralFrag) {map_highlighting=amap;}
            //ChemUtils.highlightBondsInBetweenAtoms();
        }
        //for(int za=0;za<map_highlighting.length;za++){ma.setAtomColor(map_highlighting[za],Molecule.cAtomColorRed);}
        ChemUtils.highlightBondsInBetweenAtoms(ma,ChemUtils.toBitSet(map_highlighting));
        ma.ensureHelperArrays(Molecule.cHelperCIP);
        CoordinateInventor ci = new CoordinateInventor();
        ci.invent(ma);

        return ma;
    }

}
