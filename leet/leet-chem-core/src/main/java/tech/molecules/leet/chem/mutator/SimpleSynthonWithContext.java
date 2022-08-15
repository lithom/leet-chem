package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.QueryFeatureUtils;
import tech.molecules.leet.chem.shredder.SynthonUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SimpleSynthonWithContext implements SynthonWithContext {

    @JsonPropertyDescription("synthon")
    @JsonProperty("synthon")
    private StereoMolecule synthon;
    @JsonPropertyDescription("context")
    @JsonProperty("context")private StereoMolecule context;

    @JsonPropertyDescription("map from synthon connectors to context connectors")
    @JsonProperty("connectorMap")
    private int[][] mapFromSynthonConnectorsToContextConnectors;

    public SimpleSynthonWithContext() {
        this.synthon = null;
        this.context = null;
        this.mapFromSynthonConnectorsToContextConnectors = null;
    }

    public SimpleSynthonWithContext(StereoMolecule synthon, StereoMolecule context, int[][] mapFromSynthonConnectorsToContextConnectors) {
        this.synthon = synthon;
        this.context = context;
        this.mapFromSynthonConnectorsToContextConnectors = mapFromSynthonConnectorsToContextConnectors;
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
        return this.context;
    }

    @Override
    public StereoMolecule getContextBidirectirectional(int depthInBondsSynthon, int depthInBondsContext) {
        StereoMolecule mi = new StereoMolecule();
        int[] amap_synthon = new int[this.synthon.getAtoms()];
        //int[] amap_context = new int[this.context.getAtoms()];
        //this.synthon.copyMoleculeByAtoms(mi,null,true,amap_synthon);
        //this.context.copyMoleculeByAtoms(mi,null,true,amap_context);

        mi.addFragment(this.synthon,0,amap_synthon);
        // add all context fragments..
        List<int[]> context_amaps = new ArrayList<>();
        for(int zi=0;zi<this.mapFromSynthonConnectorsToContextConnectors.length;zi++) {
            int amap_context_i[] = new int[this.context.getAtoms()];
            mi.addFragment(this.context, this.mapFromSynthonConnectorsToContextConnectors[zi][1], amap_context_i);
            context_amaps.add(amap_context_i);
        }
        mi.ensureHelperArrays(Molecule.cHelperCIP);

        // add conni/conni bonds:
        int connipos_new[][] = new int[this.mapFromSynthonConnectorsToContextConnectors.length][];
        for(int zi=0;zi<connipos_new.length;zi++) {
            int[] ccp = this.mapFromSynthonConnectorsToContextConnectors[zi];
            mi.addBond( amap_synthon[ccp[0]] , context_amaps.get(zi)[ccp[1]] , Molecule.cBondTypeSingle );
            mi.setAtomicNo( context_amaps.get(zi)[ccp[1]], 93 );
            connipos_new[zi] = new int[]{amap_synthon[ccp[0]] , context_amaps.get(zi)[ccp[1]]};
        }
        mi.ensureHelperArrays(Molecule.cHelperCIP);

        //ChemUtils.DebugOutput.plotMolecules("test1",new StereoMolecule[]{ this.synthon , this.context , mi } , 3,1 );
        return SynthonUtils.cutBidirectionalContext(mi,connipos_new,depthInBondsSynthon,depthInBondsContext,null);
    }

    @Override
    public int[][] getMapFromSynthonConnectorsToContextConnectors() {
        return this.mapFromSynthonConnectorsToContextConnectors;
    }

    @Override
    public List<int[][]> computePossibleAssemblies(SynthonWithContext other) {
        return SynthonWithContext.computeAssemblies_MatchingBondAndFirstAtom(this,other);
    }

    /**
     *
     * @param synthon
     * @param bidir_context inner connectors must be U, outer connectors must be Np
     * @return
     */
    public static List<SimpleSynthonWithContext> createAllPossibleFromSynthonAndBidirectionalContext(StereoMolecule synthon, StereoMolecule bidir_context) {
        int synthon_atommap[] = new int[synthon.getAtoms()];
        StereoMolecule ca = SynthonUtils.createConnectorProximalFragment(synthon,1, synthon_atommap);
        ca.setFragment(true);
        QueryFeatureUtils.removeNarrowingQueryFeatures(ca);
        ca.ensureHelperArrays(Molecule.cHelperCIP);

        List<Integer> connipos = ChemUtils.toIntList( SynthonUtils.findConnectorAtoms(synthon) );

        List<SimpleSynthonWithContext> all_synthons = new ArrayList<>();

        bidir_context.setFragment(true);
        QueryFeatureUtils.removeNarrowingQueryFeatures(bidir_context);
        bidir_context.ensureHelperArrays(Molecule.cHelperCIP);


        SSSearcher sss = new SSSearcher();
        //sss.setMol(ca,bidir_context);
        sss.setMolecule(bidir_context);
        sss.setFragment(ca);
        sss.findFragmentInMolecule();
        for( int mi[] : sss.getMatchList()) {
            int connimap[][] = new int[connipos.size()][2]; // [0]: pos in synthon, [1]: pos in bidir_context
            //Map<Integer,Integer> mi_inv = ChemUtils.inverseMap(mi);

            for(int zi=0;zi<connipos.size();zi++) {
                //connimap[zi] = new int[]{ connipos.get(zi) , mi_inv.get(connipos.get(zi)) };
                connimap[zi] = new int[]{ connipos.get(zi) , mi[ synthon_atommap[connipos.get(zi)] ] };
            }

            // now last thing to do: create the context molecule: i.e. for every connector, copy it with its
            // non-U neighbor into the molecule:
            StereoMolecule m_context = new StereoMolecule();
            m_context.setFragment(true);
            int connimap_final[][] = new int[connipos.size()][2];
            for(int zi=0;zi<connipos.size();zi++) {
                //int pa = mi_inv.get(connipos.get(zi)); // pos of Np conni
                int pa = connimap[zi][1];//mi[connipos.get(zi)];
                int pb = -1;
                int an_a = bidir_context.getAtomicNo( bidir_context.getConnAtom(pa,0));
                int an_b = bidir_context.getAtomicNo( bidir_context.getConnAtom(pa,1));
                if(an_a<88) {pb = bidir_context.getConnAtom(pa,0);}
                if(an_b<88) {pb = bidir_context.getConnAtom(pa,1);}

                int amap[] = new int[bidir_context.getAtoms()];
                boolean[] atoms_tc = new boolean[bidir_context.getAtoms()];
                atoms_tc[pa]=true; atoms_tc[pb]=true;
                StereoMolecule bdc_frag1 = new StereoMolecule();
                bidir_context.copyMoleculeByAtoms(bdc_frag1,atoms_tc,true,amap);
                bidir_context.ensureHelperArrays(Molecule.cHelperNeighbours);
                int amap2[] = new int[2];
                m_context.addFragment(bdc_frag1,0,amap2);
                connimap_final[zi] = new int[]{ connimap[zi][0] , amap2[amap[ connimap[zi][1] ]] };
            }

            all_synthons.add( new SimpleSynthonWithContext(synthon,m_context,connimap_final) );
        }
        return all_synthons;
    }

}
