package tech.molecules.leet.chem.shredder;


import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.coords.CoordinateInventor;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.CombinatoricsUtils;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a SplitResult that contains a central fragment,
 * in the sense that the central fragment contains half of
 * all connectors, i.e. is connected to every other
 * fragment in teh split result
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
     *
     * @return an assembled molecule, but there are the two connector atoms inbetween
     * all assembled fragments. The inner connectors are U connectors, the outer connectors
     * are Np connectors.
     *
     */
    public StereoMolecule createCombinedFragmentsMoleculeWithLinkerConnectors(int cp_pair_positions[][], boolean addConnectorConnectorBonds) {
        StereoMolecule ma = new StereoMolecule();
        List<int[]> amaps = new ArrayList<>();
        for(int zf=0;zf<splitResult.fragments.length;zf++) {
            StereoMolecule fi = splitResult.fragments[zf];
            StereoMolecule fi_copy = new StereoMolecule(fi);
            fi_copy.ensureHelperArrays(Molecule.cHelperCIP);
            int amap[] = new int[fi_copy.getAtoms()];
            ma.addFragment(fi_copy,0,amap);
            amaps.add(amap);
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
            // now search the counterpart:
            for(int zo : CombinatoricsUtils.intSeq(splitResult.fragments.length, Collections.singletonList(centralFrag))) {
                if(splitResult.connector_positions.get(zo)[zc]>=0) {
                    // ok we found it, assemble :)
                    int cfposnew = amaps.get(centralFrag)[cfpos];
                    int oposnew  = amaps.get(zo)[splitResult.connector_positions.get(zo)[zc]];

                    //System.out.println("[INFO] connect: "+cfposnew+" <-> "+oposnew);
                    if(cp_pair_positions!=null && cp_pair_positions.length>zc) { cp_pair_positions[zc] = new int[]{ cfposnew,oposnew }; }

                    // i.e. connect them..
                    if(addConnectorConnectorBonds) {
                        ma.addBond(cfposnew, oposnew, Molecule.cBondTypeSingle);
                    }
                    ma.setAtomicNo(oposnew, 93);
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

        StereoMolecule lc = this.createCombinedFragmentsMoleculeWithLinkerConnectors(cp_pair_positions,true);

        StereoMolecule all_prs = new StereoMolecule();
        all_prs.setFragment(true);

        // create fragments
        for(int zp=0;zp<cp_pair_positions.length;zp++) {
            List<Integer> seed_atoms = Arrays.stream(cp_pair_positions[zp]).boxed().collect(Collectors.toList());
            StereoMolecule mi_a = ChemUtils.createProximalFragment(lc,seed_atoms,region_size,true);
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
