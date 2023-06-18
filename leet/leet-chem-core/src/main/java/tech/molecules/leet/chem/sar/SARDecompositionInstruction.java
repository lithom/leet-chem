package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import java.util.*;


/**
 *
 * OK, here is how this works..
 *
 * The decomposition algorithm in the first step creates all StereoMolecule fragments that consist of
 * one fragment element of each multi fragment element.
 *
 * Connectors can appear once or twice, in case that they appear twice, a bridged bond (of arbitrary length) is
 * created in between the two connectors.
 * The label of a connector determines the name of the matched r-group
 *
 * Then, the decomposition works as follows: we try all assembled fragments
 *
 *
 */
public class SARDecompositionInstruction {

    public static class MultiFragment {
        private List<MultiFragmentElement> elements = new ArrayList<>();
        public List<MultiFragmentElement> getMultiFragmentElements() {return elements;}
    }

    public static class FragConnectorAtom {
        final int pos;
        final int id;
        final String name;
        public FragConnectorAtom(int pos, int id, String name) {
            this.pos = pos;
            this.id = id;
            this.name = name;
        }
    }

    public static class MultiFragmentElement {
        private String fragmentId;
        private StereoMolecule fi;

        public MultiFragmentElement(String fragmentId, StereoMolecule fi) {
            fi.ensureHelperArrays(Molecule.cHelperCIP);
            this.fragmentId = fragmentId;
            this.fi = fi;
        }
        public String getFragmentId() {
            return fragmentId;
        }
        public StereoMolecule getFi() {
            return fi;
        }
        public Set<Integer> getConnectors() {
            Set<Integer> connis = new HashSet <>();
            for(int zi=0;zi<8;zi++) {
                if(ChemUtils.countAtoms(fi, Collections.singletonList(92+zi))>0) {
                    connis.add(zi);
                }
            }
            return connis;
        }

        public List<FragConnectorAtom> getConnectorAtoms() {
            List<FragConnectorAtom> fcs = new ArrayList<>();
            for(int zi=0;zi<this.fi.getAtoms();zi++) {
                if(this.fi.getAtomicNo(zi)>=92) {
                    fcs.add(new FragConnectorAtom(zi,this.fi.getAtomicNo(zi)-92,this.fi.getAtomCustomLabel(zi)));
                }
            }
            return fcs;
        }
    }



}
