package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import java.util.List;

public class SimpleSARTests_A {

    public static void main(String args[]) {
        test_a();
    }

    public static void test_a() {
        String structure = "ejYQHL@JGOJfko`[@IABPpdLbdLRfTTRrfTtRRTt\\RxDBrV^N~qIjijjZjh@@HJjXJBQ@@";
        StereoMolecule mi = ChemUtils.parseIDCode(structure);

        String sar_structure = "ffcQ@@DQAdTRbTRbaRRtdhNTeZjj`hJJb@C[VJLyIVc^CfmBJhhRiJjk}DiaRpZJVBqRp^JWLIRhYJVBiRp]JWLYRykJUByRh_JWLERyhjUBeRh\\jWLURhZyWrE@";
        StereoMolecule sar = ChemUtils.parseIDCode(sar_structure);

        List<SimpleSARDecomposition.SimpleSARResult> results = SimpleSARDecomposition.matchSimpleSAR(sar,mi);

        System.out.println("mkay");

    }

}
