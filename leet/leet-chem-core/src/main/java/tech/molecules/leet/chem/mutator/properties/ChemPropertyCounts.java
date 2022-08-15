package tech.molecules.leet.chem.mutator.properties;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.prediction.MolecularPropertyHelper;
import tech.molecules.leet.chem.ChemUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ChemPropertyCounts {


    public static final int CHEM_PROP_COUNT_ATOMS_HEAVY      = 0;
    public static final int CHEM_PROP_COUNT_ATOMS_RING       = 1;
    public static final int CHEM_PROP_COUNT_ATOMS_NON_RING   = 2;
    public static final int CHEM_PROP_COUNT_ATOMS_AROMATIC   = 3;
    public static final int CHEM_PROP_COUNT_ATOMS_C          = 4;
    public static final int CHEM_PROP_COUNT_ATOMS_NON_C      = 5;
    public static final int CHEM_PROP_COUNT_ATOMS_HDONORS    = 6;
    public static final int CHEM_PROP_COUNT_ATOMS_NACCEPTORS = 7;
    public static final int CHEM_PROP_COUNT_ATOMS_HALOGENS   = 8;
    public static final int CHEM_PROP_COUNT_ATOMS_N          = 9;
    public static final int CHEM_PROP_COUNT_ATOMS_O          = 10;
    public static final int CHEM_PROP_COUNT_ATOMS_S          = 11;

    public static final int CHEM_PROP_COUNT_RINGS                = 20;
    public static final int CHEM_PROP_COUNT_RINGS_AROMATIC       = 21;
    public static final int CHEM_PROP_COUNT_RINGS_HETEROAROMATIC = 22;

    public static final int CHEM_PROP_COUNT_BONDS           = 30;
    public static final int CHEM_PROP_COUNT_BONDS_ROTATABLE = 31;

    public static class ChemPropertyCount {
        public final String name;
        public final Function<StereoMolecule,Integer> evaluator;
        public ChemPropertyCount(String name, Function<StereoMolecule, Integer> evaluator) {
            this.name = name;
            this.evaluator = evaluator;
        }
    }

    public static final ChemPropertyCount CountAtomsHeavy       = new ChemPropertyCount("Heavy Atom Count",(x) -> x.getAtoms());
    public static final ChemPropertyCount CountAtomsRing        = new ChemPropertyCount("Ring Atom Count",(x) -> ChemUtils.countRingAtoms(x));

    public static final ChemPropertyCount CountAtomsNonRing        = new ChemPropertyCount("Non-Ring Atom Count",(x) -> ChemUtils.countRingAtoms(x));
    public static final ChemPropertyCount CountAtomsAromatic    = new ChemPropertyCount("Aromatic Atom Count",(x) -> ChemUtils.countAromaticAtoms(x));
    public static final ChemPropertyCount CountAtomsC           = new ChemPropertyCount("C Atom Count",(x) -> ChemUtils.countAtoms(x, Collections.singletonList(6)));
    public static final ChemPropertyCount CountAtomsNonC        = new ChemPropertyCount("Non-C Atom Count",(x) -> x.getAtoms()-ChemUtils.countAtoms(x, Collections.singletonList(6)));

    public static final ChemPropertyCount CountAtomsHDonors     = new ChemPropertyCount("H-Donor Atom Count",(x) -> (int) MolecularPropertyHelper.calculateProperty(x,MolecularPropertyHelper.MOLECULAR_PROPERTY_HDONORS));
    public static final ChemPropertyCount CountAtomsHAcceptors  = new ChemPropertyCount("H-Acceptor Atom Count",(x) -> (int) MolecularPropertyHelper.calculateProperty(x,MolecularPropertyHelper.MOLECULAR_PROPERTY_HACCEPTORS));

    public static final ChemPropertyCount CountAtomsHalogens    = new ChemPropertyCount("Halogen Atom Count",(x) -> ChemUtils.countAtoms(x, Arrays.stream( new int[]{9,17,35,53} ).boxed().collect(Collectors.toList()) ));
    public static final ChemPropertyCount CountAtomsN           = new ChemPropertyCount("N Atom Count",(x) -> ChemUtils.countAtoms(x, Collections.singletonList(7)));
    public static final ChemPropertyCount CountAtomsO           = new ChemPropertyCount("O Atom Count",(x) -> ChemUtils.countAtoms(x, Collections.singletonList(8)));
    public static final ChemPropertyCount CountAtomsS           = new ChemPropertyCount("S Atom Count",(x) -> ChemUtils.countAtoms(x, Collections.singletonList(16)));


    public static final ChemPropertyCount CountRings                = new ChemPropertyCount("Ring Count",(x) -> ChemUtils.countRings(x));
    public static final ChemPropertyCount CountAromaticRings        = new ChemPropertyCount("Aromatic Ring Count",(x) -> ChemUtils.countRingsAromatic(x) );
    public static final ChemPropertyCount CountHeteroaromaticRings  = new ChemPropertyCount("Heteroaromatic Ring Count",(x) -> x.getAtoms()-ChemUtils.countRingsHeteroaromatic(x) );

    public static final ChemPropertyCount CountBonds                = new ChemPropertyCount("Bond Count",(x) -> x.getBonds());
    public static final ChemPropertyCount CountRotatableBonds       = new ChemPropertyCount("Rotatable Bond Count",(x) -> x.getRotatableBondCount());



    public static final ChemPropertyCount[] COUNTS_ALL = new ChemPropertyCount[]{
            CountAtomsHeavy, CountAtomsRing, CountAtomsNonRing, CountAtomsAromatic, CountAtomsC, CountAtomsNonC,
            CountAtomsHDonors, CountAtomsHAcceptors,

            CountAtomsHalogens, CountAtomsN, CountAtomsO, CountAtomsS,
            CountRings, CountAromaticRings, CountHeteroaromaticRings,

            CountRotatableBonds   /**CountBonds,*/
    };

}
