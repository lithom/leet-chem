package tech.molecules.leet.chem.injector;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;
import tech.molecules.leet.chem.shredder.SynthonShredder;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class InjectorTools {


    /**
     *
     * Creates the fragment decomposition, based on the selection of a "central fragment".
     * In case that the selection does not cover a central fragment an excpetion is thrown.
     *
     * @param mi
     * @return
     */
    public static FragmentDecomposition computeFragmentDecompositionFromSelection(StereoMolecule mi) throws InvalidSelectionException {
        mi.ensureHelperArrays(Molecule.cHelperCIP);

        List<Integer> bonds_to_cut = new ArrayList<>();
        for(int zi=0;zi<mi.getBonds();zi++) {
            int ba = mi.getBondAtom(0,zi);
            int bb = mi.getBondAtom(1,zi);
            boolean ba_in = mi.isSelectedAtom(ba);
            boolean bb_in = mi.isSelectedAtom(bb);

            if(ba_in ^ bb_in) {
                bonds_to_cut.add(zi);
            }
        }

        // now remove bonds..
        StereoMolecule m2 = new StereoMolecule(mi);
        m2.ensureHelperArrays(Molecule.cHelperCIP);

        for(int zi=0;zi<bonds_to_cut.size();zi++) {
            m2.markBondForDeletion(zi);
        }

        int[] amap = m2.deleteMarkedAtomsAndBonds();

        List<Integer> fragatoms  = ChemUtils.getSelectedAtoms(mi);
        List<Integer> fragatoms2 = fragatoms.stream().mapToInt( fi -> amap[fi] ).boxed().collect(Collectors.toList());

        boolean same_frag = ChemUtils.checkIfAllAtomsAreInSameFragment(m2,fragatoms2);

        if(!same_frag) {
            throw new InvalidSelectionException("Invalid selection containing unconnected atoms");
        }

        // compute split result..
        SynthonShredder.SplitResult sr = SynthonShredder.trySplit(mi,ChemUtils.toIntArray(bonds_to_cut),10);
        if(sr==null) {
            throw new InvalidSelectionException("Invalid selection, split did not work..");
        }

        // find central resulting fragment:
        int central_frag = sr.fragment_positions[fragatoms2.get(0)];
        // sanity check: all fragment_positions the same for all fragatoms2?:
        if(true) {
            boolean sanity_a = true;
            for (int pa : fragatoms2){
                sanity_a &= sr.fragment_positions[pa]==central_frag;
                if(!sanity_a) {throw new Error("fragment positions wrong.. :(");}
            }
        }

        FragmentDecomposition fd = new FragmentDecomposition("test",sr,central_frag);
        return fd;
    }

    public static class InvalidSelectionException extends Exception {
        public InvalidSelectionException(String msg) {
            super(msg);
        }
    }

}
