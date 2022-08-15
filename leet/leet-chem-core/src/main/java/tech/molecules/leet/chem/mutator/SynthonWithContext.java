package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.synthon.SynthonReactor;
import tech.molecules.leet.chem.shredder.SynthonUtils;

import java.util.ArrayList;
import java.util.List;

public interface SynthonWithContext {

    public StereoMolecule getSynthon();
    public StereoMolecule getContext();

    public StereoMolecule getContext(int depthInBonds);

    public StereoMolecule getContextBidirectirectional(int depthInBondsSynthon, int depthInBondsContext);

    public int[][] getMapFromSynthonConnectorsToContextConnectors();

    /**
     *
     * @param other
     * @return map from this synthon molecule connectors to other synthon molecule connectors.
     */
    public List<int[][]> computePossibleAssemblies(SynthonWithContext other);


    public static List<int[][]> computeAssemblies_MatchingBondAndFirstAtom(SynthonWithContext a, SynthonWithContext b) {

        int map_a[] = new int[a.getContext().getAtoms()];
        int map_b[] = new int[b.getContext().getAtoms()];

        StereoMolecule ca = SynthonUtils.createConnectorProximalFragment(a.getContext(),1,map_a);
        StereoMolecule cb = SynthonUtils.createConnectorProximalFragment(b.getContext(),1,map_b);

        SSSearcher ss = new SSSearcher();
        ss.setMol(ca,cb);
        ss.findFragmentInMolecule();
        ArrayList<int[]> matches = ss.getMatchList();

        List<int[][]> assemblies = new ArrayList<>();
        for(int[] mi : matches) {
            int[][] assembly_i = new int[a.getMapFromSynthonConnectorsToContextConnectors().length][];
            for(int zs=0;zs<assembly_i.length;zs++) {
                int[] ci = a.getMapFromSynthonConnectorsToContextConnectors()[zs];
                //find matched conni in b:
                int found = -1;
                for(int zi=0;zi<mi.length;zi++) {
                    if( ci[0] == mi[zi]) { found = zi; break; }
                }
                if(found<0) {throw new Error("Something wrong..");}

                // now connect b:found and a:ci[0]
                assembly_i[zs] = new int[]{ ci[0] , found };
            }
            assemblies.add(assembly_i);
        }
        return assemblies;
    }

    /**
     *
     * @param a
     * @param b
     * @param assembly
     * @return
     */
    public static StereoMolecule annealSynthons(SynthonWithContext a, SynthonWithContext b, int[][] assembly) {
        //boolean is_fragment = a.getSynthon().isFragment() || b.getSynthon().isFragment();
        //StereoMolecule ma = new StereoMolecule();
        //ma.setFragment(is_fragment);

        //int pa[] = new int[a.getSynthon().getAtoms()];
        //int pb[] = new int[b.getSynthon().getAtoms()];
        //ma.addFragment(a.getSynthon(),0,pa);
        //ma.addFragment(b.getSynthon(),0,pb);

        StereoMolecule a1 = new StereoMolecule(a.getSynthon());
        a1.ensureHelperArrays(Molecule.cHelperCIP);
        StereoMolecule a2 = new StereoMolecule(b.getSynthon());
        a2.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<assembly.length;zi++) {
            a1.setAtomicNo(assembly[zi][0],92+zi);
            a2.setAtomicNo(assembly[zi][1],92+zi);
        }

        List<StereoMolecule> frags = new ArrayList<>();
        frags.add(a1);
        frags.add(a2);

        StereoMolecule assembled = SynthonReactor.react(frags);
        assembled.ensureHelperArrays(Molecule.cHelperCIP);
        return assembled;
    }

}
