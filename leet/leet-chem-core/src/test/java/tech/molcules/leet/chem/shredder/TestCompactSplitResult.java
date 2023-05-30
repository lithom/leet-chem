package tech.molcules.leet.chem.shredder;

import com.actelion.research.chem.MoleculeStandardizer;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.synthon.SynthonReactor;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.shredder.SynthonShredder;

import java.util.Arrays;
import java.util.List;

public class TestCompactSplitResult {

    public static void main(String args[]) {
        testCompactSplitResult_01();
    }

    /**
     * For cutting apart aromatic rings the split result sometimes is not absolutely equal,
     * but this should not matter in real world scenarios.
     *
     */
    public static void testCompactSplitResult_01() {
        String idc = "ejY^H@@DGLbjgdPDfVYU^uVyU[{WTlL|fnv^qIZ`@@Hj`hJbB@@`@@";


        StereoMolecule mi = ChemUtils.parseIDCode(idc);

        List<SynthonShredder.SplitResult> splits = SynthonShredder.computeAllSplitResults(mi,3,4,true);

        boolean all_ok = true;
        for(int zi=0;zi<splits.size();zi++) {
            SynthonShredder.SplitResult sri         = splits.get(zi);

            for(int zj=0;zj<sri.fragments.length;zj++) {
                try {
                    MoleculeStandardizer.standardize(sri.fragments[zj], 0);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            SynthonShredder.CompactUnlabeledSplitResult csri = SynthonShredder.convertToUnlabeledCompactSplitResult(sri);
            SynthonShredder.SplitResult sri2        = SynthonShredder.convert(csri);
            boolean ok_i = sri.toString().equals(sri2.toString());
            System.out.println("sri1 equals to sri2: " + ok_i );
            if(!ok_i) {
                System.out.println( sri.toString() + "\n"+ sri2.toString() );
                for(int zj=0;zj<sri.fragments.length;zj++) {
                    try {
                        MoleculeStandardizer.standardize(sri.fragments[zj], 0);
                        MoleculeStandardizer.standardize(sri2.fragments[zj], 0);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    String idc_a = sri.fragments[zj].getIDCode();
                    String idc_b = sri2.fragments[zj].getIDCode();
                    System.out.println(idc_a);
                    System.out.println(idc_b);

                    SSSearcher sss1 = new SSSearcher();
                    StereoMolecule fi = new StereoMolecule(sri.fragments[zj]);
                    fi.setFragment(true);
                    sss1.setMol(fi,sri2.fragments[zj]);
                    boolean ok_ss_1 = sss1.isFragmentInMolecule();

                    SSSearcher sss2 = new SSSearcher();
                    StereoMolecule fi2 = new StereoMolecule(sri2.fragments[zj]);
                    fi2.setFragment(true);
                    sss2.setMol(fi2,sri.fragments[zj]);
                    boolean ok_ss_2 = sss2.isFragmentInMolecule();
                    System.out.println("mkay: ok1: "+ok_ss_1+"  ok2: "+ok_ss_2);
                }
                //System.out.println("mkay");
                //StereoMolecule mas_0 = SynthonReactor.react( Arrays.asList( sri2.fragments ) );
                //System.out.println("assembled: "+mas_0.getIDCode());
                //System.out.println("original:  "+mi.getIDCode());
                //System.out.println("mkay");
                //
                //StereoMolecule mas_1 = SynthonReactor.react( Arrays.asList( sri2.fragments ) );
                //String idc0 = mas_0.getIDCode();
                //String idc1 = mas_1.getIDCode();
//                System.out.println(sri.toString());
//                System.out.println(sri2.toString());
//                System.out.println(idc0);
//                System.out.println(idc1);
//                System.out.println("mkay..");
            }
            all_ok &= ok_i;
        }
        System.out.println("all ok: "+all_ok);
    }

}
