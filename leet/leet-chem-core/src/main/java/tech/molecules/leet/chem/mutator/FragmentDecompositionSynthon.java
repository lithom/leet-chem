package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;
import tech.molecules.leet.chem.shredder.SynthonUtils;

import java.util.ArrayList;
import java.util.List;

public class FragmentDecompositionSynthon implements SynthonWithContext {

    @JsonPropertyDescription("underlying fragment decomposition")
    @JsonProperty("decomp")
    private FragmentDecomposition decomp;

    @JsonPropertyDescription("synthon")
    @JsonProperty("synthon")
    private StereoMolecule synthon;

    @JsonPropertyDescription("synthon_idcode")
    @JsonProperty("synthonIDCode")
    private String synthonIDCode;

    @JsonPropertyDescription("remainder_idcode")
    @JsonProperty("remainderIDCode")
    private String remainderIDCode;

    @JsonPropertyDescription("context")
    @JsonProperty("context")
    private StereoMolecule context;
    @JsonPropertyDescription("map from synthon connectors to context connectors")
    @JsonProperty("connectorMap")
    private int[][] mapSynthonConnectorsToContextConnectors;


    public FragmentDecompositionSynthon() {}

    public FragmentDecompositionSynthon(FragmentDecomposition decomp) {
        //this.synthon = new StereoMolecule( decomp.getCentralFrag() );
        int cpos[][] = new int[decomp.getNumberOfConnectors()][2];
        List<int[]> frag_amaps = new ArrayList<>();
        StereoMolecule combined = decomp.createCombinedFragmentsMoleculeWithLinkerConnectors(cpos,false,frag_amaps);
        this.synthon = new StereoMolecule();
        this.synthon.setFragment(combined.isFragment());

        // central: only one add fragment needed, for the context we have to separately copy all fragments
        int map_central[] = new int[combined.getAtoms()];
        this.synthon.addFragment(combined,cpos[0][0],map_central);
        this.synthon.ensureHelperArrays(Molecule.cHelperCIP);

        this.context = new StereoMolecule();
        this.context.setFragment(combined.isFragment());
        List<int[]> maps_context = new ArrayList<>();
        for(int zi=0;zi<cpos.length;zi++) {
            int map_cont_i[] = new int[combined.getAtoms()];
            this.context.addFragment(combined,cpos[zi][1],map_cont_i);
            maps_context.add(map_cont_i);
        }
        this.context.ensureHelperArrays(Molecule.cHelperCIP);

        this.mapSynthonConnectorsToContextConnectors = new int[cpos.length][2];
        for(int ci = 0;ci<cpos.length;ci++) {
            int pos_a = map_central[cpos[ci][0]];
            int pos_b = maps_context.get(ci)[cpos[ci][1]];
            this.mapSynthonConnectorsToContextConnectors[ci][0] = pos_a;
            this.mapSynthonConnectorsToContextConnectors[ci][1] = pos_b;
        }
        this.decomp = decomp;
        this.setSynthonIDcode(this.synthon.getIDCode());
        this.setRemainderIDCode(this.context.getIDCode());
    }

    @Override
    public StereoMolecule getSynthon() {
        return this.synthon;
    }

    @Override
    public StereoMolecule getContext() {
        return this.context;
    }

    @Override
    public StereoMolecule getContext(int depthInBonds) {
        StereoMolecule mi = new StereoMolecule();
        mi.ensureHelperArrays(Molecule.cHelperCIP);

        StereoMolecule ci = ChemUtils.createProximalFragment(mi, ChemUtils.toIntList(SynthonUtils.findConnectorAtoms(mi)), depthInBonds,false,null);
        ci.ensureHelperArrays(Molecule.cHelperCIP);
        return ci;
    }

    @Override
    public StereoMolecule getContextBidirectirectional(int depthInBondsSynthon, int depthInBondsContext) {
        // not like this, instead we do this separately for every connector
        //return SynthonUtils.createConnectorProximalFragment(this.createCombinedFragmentsMoleculeWithLinkerConnectors(),region_size);
        int[][] cp_pair_positions = new int[this.decomp.getSplitResult().connector_positions.get(0).length][2];

        List<int[]> sr_atom_maps = new ArrayList<>();
        StereoMolecule lc = this.decomp.createCombinedFragmentsMoleculeWithLinkerConnectors(cp_pair_positions,true, sr_atom_maps);

        return SynthonUtils.cutBidirectionalContext(lc,cp_pair_positions,depthInBondsSynthon,depthInBondsContext,null);
    }

    @Override
    public int[][] getMapFromSynthonConnectorsToContextConnectors() {
        return this.mapSynthonConnectorsToContextConnectors;
    }

    @Override
    public List<int[][]> computePossibleAssemblies(SynthonWithContext other) {
        return SynthonWithContext.computeAssemblies_MatchingBondAndFirstAtom(this,other);
    }

    public String getSynthonIDCode() {
        return synthonIDCode;
    }

    public void setSynthonIDcode(String synthonIDcode) {
        this.synthonIDCode = synthonIDcode;
    }

    public String getRemainderIDCode() {
        return remainderIDCode;
    }

    public void setRemainderIDCode(String remainderIDCode) {
        this.remainderIDCode = remainderIDCode;
    }
}
