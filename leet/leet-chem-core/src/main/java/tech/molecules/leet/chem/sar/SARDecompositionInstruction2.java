package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.shredder.SynthonShredder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 *
 * OK, here is how this works..
 *
 * A scaffold consists of multiple multifragments.
 * Each fragment in a multifragment has atoms that are "extendable" (this is indicated by a label).
 * NOTE FOR MULTIFRAGMENTS: the idea is that the "same" extendable atoms are labeled in the same way..
 *
 * Then, the decomposition algorithm builds all combinations of multifragments and performs for each the following steps:
 *
 * 1. all fragments are copied into one stereomolecule (fragment)
 * 2. all non-labeled atoms are set to "prevent further substitution"
 * 3. then the project structure is matched (in case of multiple matchings a warning is issued)
 * 4. we split the structure with synthonlogic as follows: all bonds "extendable"-"other scaffold atom" are cut and
 *    a connector is put in there
 * 5. all parts of the structure that consist only of scaffold atoms (and connectors) are removed
 * 6. the remaining connected graphs are the "r-groups", each fragment is assigned to the set of
 *    extendable atoms that it contains. The labels of the extendable atoms remain.
 *
 *
 * Now, reassembling molecules should be easy: the extendable atoms of the scaffold and the ones in the fragment
 * are "aligned", i.e. combined into one atom that contains both all scaffold and r-group bonds.
 *
 *
 */
public class SARDecompositionInstruction2 {




    public static void matchSARElements( List<SARElement> sar_scaffold , List<StereoMolecule> mols) {
        StereoMolecule qf = new StereoMolecule();
        qf.setFragment(true);
        for(SARElement ei : sar_scaffold) {
            qf.addFragment(ei.getQueryFrag(),0,null);
        }
        qf.ensureHelperArrays(Molecule.cHelperCIP);
        //System.out.println("qf:" + (new Canonizer(qf,Canonizer.ENCODE_ATOM_CUSTOM_LABELS).getIDCode()));

        for(StereoMolecule mi : mols) {
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            matchSAR(qf,mi);
        }
    }

    public static void matchSAR(StereoMolecule sfrag, StereoMolecule mol_a) {
        SSSearcher sss = new SSSearcher();
        sss.setMol(sfrag,mol_a);
        sss.findFragmentInMolecule();

        List<int[]> matches = sss.getMatchList();
        System.out.println("matches: "+matches.size());

        // now determine r groups for matches:
        for(int[] mi : matches) {

            StereoMolecule mol = new StereoMolecule(mol_a);
            mol.ensureHelperArrays(Molecule.cHelperCIP);
            // map labels..
            for(int zi=0;zi<mi.length;zi++) {
                mol.setAtomCustomLabel(mi[zi],sfrag.getAtomCustomLabel(zi));
            }

            // 1. find all matched atoms, they are "scaffold atoms"
            Set<Integer> scaffold_atoms = Arrays.stream(mi).boxed().collect(Collectors.toSet());

            // 1b. add all atoms that are part of bonds that are matched to bridge bonds:
            // Ah maybe we actually do not need this at all: it works without this..

//            boolean nonBridgeAtoms[] = new boolean[mol_a.getAtoms()];
//            for(int zi=0;zi<mi.length;zi++) {nonBridgeAtoms[ mi[zi] ]=true;}
//            for(int zi=0;zi<sfrag.getBonds();zi++) {
//                if(sfrag.isBondBridge(zi)) {
//                    // for the time being, these atoms will just generate a separate r-group / a part of an r-group
//                    System.out.println("not yet supported..");
//                    //sfrag
//                }
//            }

            // 2. collect all bonds (scaffold(labeled)-scaffold(any)) for splitting
            List<Integer> bonds_scaffold_scaffold = new ArrayList<>();

            int[] mi_inv = ChemUtils.inverseMap2(mi,mi.length,mol.getAtoms());

            for(int bi = 0; bi<mol.getBonds(); bi++) {
                int ba1 = mol.getBondAtom(0, bi);
                int ba2 = mol.getBondAtom(1, bi);
                int num_scaffold = (scaffold_atoms.contains(ba1) ? 1 : 0) + (scaffold_atoms.contains(ba2) ? 1 : 0);
                int num_extendable = 0;

                if (mol.getAtomCustomLabel(ba1) != null && !mol.getAtomCustomLabel(ba1).equals("")) {
                    num_extendable++;
                }
                if (mol.getAtomCustomLabel(ba2) != null && !mol.getAtomCustomLabel(ba2).equals("")) {
                    num_extendable++;
                }

                /**
                if( mi_inv[ba1] >= 0) {
                    int x_ba1 = mi_inv[ba1];
                    if( sfrag.getAtomCustomLabel(x_ba1) != null && !sfrag.getAtomCustomLabel(x_ba1).equals("") ) {
                        num_extendable++;
                    }
                }
                if( mi_inv[ba2] >= 0) {
                    int x_ba2 = mi_inv[ba2];
                    if( sfrag.getAtomCustomLabel(x_ba2) != null && !sfrag.getAtomCustomLabel(x_ba2).equals("") ) {
                        num_extendable++;
                    }
                } */

                System.out.println("nscf: "+num_scaffold+ "   n_xt: "+num_extendable);
                if(num_scaffold == 2 && num_extendable >= 1) { bonds_scaffold_scaffold.add(bi);}
            }

            int[] bss = new int[bonds_scaffold_scaffold.size()];
            for(int zi=0;zi<bss.length;zi++) {bss[zi] = bonds_scaffold_scaffold.get(zi);}

            SynthonShredder.SplitResult sri = SynthonShredder.trySplit(mol,bss,16);

            System.out.println("fragments:");
            for(StereoMolecule rri : sri.fragments) {
                System.out.println( (new Canonizer(rri,Canonizer.ENCODE_ATOM_CUSTOM_LABELS)).getIDCode() );
            }
        }
    }


    public static void main(String args[]) {
        String mol_idcode = "efuuN@@DCMh\\ZV[XtpMGBecm\\bbTTRTRaTRTRrbbvjTRRfTlRTQbg`ssOLjh\\[TE@uDETDuEUTmS@AUAQ`Q@dt`@";
        String sar_a      = "";
        String sar_b      = "";

        StereoMolecule mol = new StereoMolecule();

        //StereoMolecule sar_a = new StereoMolecule();
        //StereoMolecule sar_b = new StereoMolecule();
    }


}
